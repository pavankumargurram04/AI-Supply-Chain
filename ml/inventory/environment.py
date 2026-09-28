"""Inventory DQN environment.

Consumes real forecasts + real supplier state. Action space is a set of
*multiples of the expected lead-time demand*, which makes the policy
scale-invariant across products and warehouses.

State (per step):
    [ avail_norm, inbound_norm, forecast_mean_norm, forecast_std_norm,
      lt_mean_norm, lt_std_norm, supplier_risk, service_target,
      holding_cost_norm, shortage_cost_norm, days_since_order_norm ]

Action (discrete):
    index into `action_multipliers` (default: 0.0, 0.25, 0.5, ..., 2.5)
    quantity = multiplier * expected_lead_time_demand

Reward:
    -(holding + shortage + ordering) for the step, normalised by a scale
    so that RL training is stable across product sizes.

Episode:
    walks through a demand sequence of length T_start..T_end, updating
    inventory on every step, applying lead-time delay to inbound orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.deterministic import derive_seed, rng_from_seed
from ml.inventory.cost_model import CostParameters


@dataclass
class EnvConfig:
    # Action space — relative to expected lead-time demand.
    action_multipliers: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)

    # Episode length (in days).
    episode_days: int = 60

    # Reward normalisation scale. Reward is divided by this.
    reward_scale: float = 100.0

    # Normalisation constants used in the state vector.
    inv_norm: float = 500.0
    forecast_norm: float = 100.0
    lead_time_norm: float = 30.0

    # Cost parameters are supplied per instance (from the product).
    cost: Optional[CostParameters] = None

    # Review period in days (1 = daily review).
    review_period: int = 1

    # Max observations to build the state vector length.
    state_dim: int = 11
    n_actions: int = 9

    def to_dict(self) -> dict:
        return {
            "action_multipliers": list(self.action_multipliers),
            "episode_days": self.episode_days,
            "reward_scale": self.reward_scale,
            "inv_norm": self.inv_norm,
            "forecast_norm": self.forecast_norm,
            "lead_time_norm": self.lead_time_norm,
            "review_period": self.review_period,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "cost": self.cost.to_dict() if self.cost else None,
        }


class InventoryEnvironment:
    """Single-series environment for one (product, warehouse) pair."""

    def __init__(self, cfg: EnvConfig) -> None:
        if cfg.cost is None:
            raise ValueError("EnvConfig.cost is required.")
        self.cfg = cfg
        self.n_actions = len(cfg.action_multipliers)

    # -- Setup a series from pre-computed arrays ------------------------
    def attach_series(
        self,
        *,
        forecast_point: np.ndarray,          # (T,) mean forecast per day
        forecast_std: np.ndarray,            # (T,) forecast uncertainty
        actual_demand: np.ndarray,           # (T,) realised demand
        lead_time_mean: np.ndarray,          # (T,) mean lead time per day
        lead_time_std: np.ndarray,           # (T,) std of lead time
        supplier_risk: np.ndarray,           # (T,) risk prob per day
        initial_inventory: float,
        initial_inbound: float = 0.0,
        service_target: float = 0.95,
        seed: int = 42,
    ) -> None:
        """Attach data arrays of equal length T ≥ episode_days + max_lead_time."""
        T = forecast_point.size
        for arr, name in (
            (forecast_std, "forecast_std"),
            (actual_demand, "actual_demand"),
            (lead_time_mean, "lead_time_mean"),
            (lead_time_std, "lead_time_std"),
            (supplier_risk, "supplier_risk"),
        ):
            if arr.size != T:
                raise ValueError(f"{name} must have length {T}")
        self.forecast_point = np.asarray(forecast_point, dtype=np.float32)
        self.forecast_std = np.asarray(forecast_std, dtype=np.float32)
        self.actual_demand = np.asarray(actual_demand, dtype=np.float32)
        self.lead_time_mean = np.asarray(lead_time_mean, dtype=np.float32)
        self.lead_time_std = np.asarray(lead_time_std, dtype=np.float32)
        self.supplier_risk = np.asarray(supplier_risk, dtype=np.float32)
        self.initial_inventory = float(initial_inventory)
        self.initial_inbound = float(initial_inbound)
        self.service_target = float(service_target)
        self.seed = seed

    # -- Episode management --------------------------------------------
    def reset(self, *, start_index: int = 0) -> np.ndarray:
        """Reset to the given start index; returns the initial state vector."""
        self.t = int(start_index)
        self.start_index = int(start_index)
        self.on_hand = self.initial_inventory
        self.reserved = 0.0
        self.inbound = self.initial_inbound
        self.pending_orders: List[Tuple[int, float]] = []  # (arrival_day, qty)
        self.days_since_order = 99  # sentinel for "no order yet"
        self.episode_reward = 0.0
        self.episode_cost = 0.0
        self.episode_unmet = 0.0
        self.episode_demand = 0.0
        self.episode_holding_days = 0.0  # sum of on-hand over days (for holding cost)
        self.n_orders = 0
        self.rng = rng_from_seed(derive_seed(self.seed, "env", self.start_index))
        return self._state()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advance one review period."""
        if self.t >= self.forecast_point.size:
            raise IndexError("Environment has run past the end of the series.")

        # 1) Arrivals.
        arrivals = [q for (a, q) in self.pending_orders if a <= self.t]
        if arrivals:
            self.on_hand += float(sum(arrivals))
            self.inbound -= float(sum(arrivals))
            self.pending_orders = [(a, q) for (a, q) in self.pending_orders if a > self.t]

        # 2) Place order if action > 0.
        order_qty = 0.0
        expected_ltd = float(self.forecast_point[self.t] * self.lead_time_mean[self.t])
        mult = self.cfg.action_multipliers[int(action_idx)]
        if mult > 0.0:
            order_qty = mult * expected_ltd
            # Lead time sampled for this order (integer days).
            lt_sample = max(
                1,
                int(round(self.rng.normal(
                    self.lead_time_mean[self.t], max(self.lead_time_std[self.t], 0.5)
                ))),
            )
            arrival_day = self.t + lt_sample
            self.pending_orders.append((arrival_day, order_qty))
            self.inbound += order_qty
            self.n_orders += 1
            self.days_since_order = 0
        else:
            self.days_since_order = min(self.days_since_order + 1, 999)

        # 3) Realised demand this period.
        demand = float(self.actual_demand[self.t])

        # 4) Fill.
        fillable = min(self.on_hand, demand)
        self.on_hand -= fillable
        unmet = demand - fillable
        self.episode_demand += demand
        if unmet > 0:
            self.episode_unmet += unmet

        # 5) Cost for this step.
        params = self.cfg.cost
        h_cost = self.on_hand * params.unit_cost * params.holding_rate_daily
        s_cost = unmet * params.shortage_cost_per_unit
        # Order cost is applied at the moment of ordering, not at arrival.
        o_cost = (params.order_fixed_cost + params.shipping_cost_per_order) if order_qty > 0 else 0.0
        step_cost = h_cost + s_cost + o_cost
        self.episode_cost += step_cost
        self.episode_holding_days += self.on_hand

        # 6) Reward = −step_cost / scale.
        reward = -step_cost / self.cfg.reward_scale
        self.episode_reward += reward

        # 7) Advance.
        self.t += self.cfg.review_period
        done = (self.t - self.start_index) >= self.cfg.episode_days

        info = {
            "t": self.t,
            "on_hand": float(self.on_hand),
            "inbound": float(self.inbound),
            "unmet": float(unmet),
            "order_qty": float(order_qty),
            "step_cost": float(step_cost),
            "days_since_order": self.days_since_order,
        }
        return self._state(), float(reward), bool(done), info

    # -- State vector --------------------------------------------------
    def _state(self) -> np.ndarray:
        t = min(self.t, self.forecast_point.size - 1)
        fc_mean = float(self.forecast_point[t])
        fc_std = float(self.forecast_std[t])
        lt_mean = float(self.lead_time_mean[t])
        lt_std = float(self.lead_time_std[t])
        risk = float(self.supplier_risk[t])
        avail = self.on_hand - self.reserved
        params = self.cfg.cost

        s = np.array([
            avail / self.cfg.inv_norm,
            self.inbound / self.cfg.inv_norm,
            fc_mean / self.cfg.forecast_norm,
            fc_std / self.cfg.forecast_norm,
            lt_mean / self.cfg.lead_time_norm,
            lt_std / self.cfg.lead_time_norm,
            risk,
            self.service_target,
            params.holding_rate_annual,
            params.shortage_cost_per_unit / max(params.unit_cost, 1e-6),
            min(self.days_since_order, 90) / 90.0,
        ], dtype=np.float32)
        return s

    # -- Helpers -------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return 11

    def episode_summary(self) -> dict:
        params = self.cfg.cost
        service_level = 1.0 - (self.episode_unmet / max(self.episode_demand, 1e-6))
        avg_inv = self.episode_holding_days / max(self.cfg.episode_days, 1)
        return {
            "total_cost": float(self.episode_cost),
            "n_orders": int(self.n_orders),
            "unmet_units": float(self.episode_unmet),
            "demand_units": float(self.episode_demand),
            "service_level": float(service_level),
            "avg_inventory": float(avg_inv),
            "episode_days": int(self.cfg.episode_days),
            "reward": float(self.episode_reward),
            "cost_params": params.to_dict(),
        }