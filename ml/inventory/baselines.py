"""Baseline inventory policies for ablation.

Each baseline shares the same `InventoryEnvironment`. The baseline decides
an order quantity each step according to a classical rule:

    - NeverOrder:        always 0.
    - EconomicOrderQty:  Wilson EOQ based on forecast mean demand.
    - sS:                periodic (s, S) policy.
    - RQ:                (R, Q) — reorder when inventory position < R, order Q.
    - Newsvendor:        order the α-quantile of lead-time demand each review.

Baselines consume the same forecast + lead-time state as the DQN, so
comparisons are apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ml.inventory.cost_model import CostParameters
from ml.inventory.environment import EnvConfig, InventoryEnvironment


# ---------------------------------------------------------------------------
# Each baseline takes a state + expected lead-time demand and returns an
# action index into the environment's action_multipliers.
# ---------------------------------------------------------------------------
def _multiplier_to_index(env: InventoryEnvironment, mult: float) -> int:
    mults = np.asarray(env.cfg.action_multipliers)
    idx = int(np.argmin(np.abs(mults - mult)))
    return idx


def never_order(env: InventoryEnvironment, state: np.ndarray, expected_ltd: float) -> int:
    return _multiplier_to_index(env, 0.0)


def newsvendor(
    env: InventoryEnvironment,
    state: np.ndarray,
    expected_ltd: float,
    *,
    critical_ratio: float,
) -> int:
    """Order up to the α-quantile of lead-time demand.

    We approximate the quantile by (1 + z) * expected_ltd where z is a
    linearisation of the critical ratio. For an exact quantile we would need
    the full distribution, which the DQN environment only exposes via the
    forecast mean. This is a *classical* baseline — its simplicity is the point.
    """
    # Simple approximation: α=0.5 → 1.0×, α=0.95 → 1.6×, α=0.99 → 2.0×.
    if critical_ratio <= 0.5:
        mult = 1.0
    elif critical_ratio <= 0.75:
        mult = 1.15
    elif critical_ratio <= 0.9:
        mult = 1.35
    elif critical_ratio <= 0.95:
        mult = 1.6
    else:
        mult = 2.0
    return _multiplier_to_index(env, mult)


def economic_order_quantity(
    env: InventoryEnvironment,
    state: np.ndarray,
    expected_ltd: float,
    *,
    annual_demand: float,
) -> int:
    """Wilson EOQ → order quantity → multiplier.

    Q* = sqrt(2 * D * K / h)
    where D = annual demand, K = fixed order cost, h = holding cost per unit per year.
    """
    params = env.cfg.cost
    if params is None:
        return _multiplier_to_index(env, 0.0)
    h = params.unit_cost * params.holding_rate_annual
    if h <= 0 or annual_demand <= 0:
        return _multiplier_to_index(env, 0.0)
    Q_star = np.sqrt(2.0 * annual_demand * params.order_fixed_cost / h)
    mult = float(Q_star) / max(expected_ltd, 1e-6)
    # Clip to the action space.
    mult = float(np.clip(mult, 0.0, max(env.cfg.action_multipliers)))
    return _multiplier_to_index(env, mult)


def sS_policy(
    env: InventoryEnvironment,
    state: np.ndarray,
    expected_ltd: float,
    *,
    s_mult: float = 0.8,
    S_mult: float = 1.5,
) -> int:
    """Order up to S if inventory position < s.

    Both s and S are expressed as multiples of expected lead-time demand.
    """
    inv_position = env.on_hand + env.inbound - env.reserved
    s = s_mult * expected_ltd
    S = S_mult * expected_ltd
    if inv_position < s:
        qty = max(0.0, S - inv_position)
        mult = qty / max(expected_ltd, 1e-6)
        return _multiplier_to_index(env, float(np.clip(mult, 0.0, max(env.cfg.action_multipliers))))
    return _multiplier_to_index(env, 0.0)


def RQ_policy(
    env: InventoryEnvironment,
    state: np.ndarray,
    expected_ltd: float,
    *,
    R_mult: float = 0.8,
    Q_mult: float = 1.2,
) -> int:
    inv_position = env.on_hand + env.inbound - env.reserved
    R = R_mult * expected_ltd
    if inv_position < R:
        return _multiplier_to_index(env, Q_mult)
    return _multiplier_to_index(env, 0.0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class BaselineResult:
    name: str
    total_cost: float
    service_level: float
    unmet_units: float
    n_orders: int
    avg_inventory: float
    reward: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def run_baseline(
    name: str,
    series: dict,
    *,
    cost: CostParameters,
    episode_days: int = 60,
    start_index: int = 0,
    seed: int = 42,
) -> BaselineResult:
    """Run one baseline on one series."""
    env_cfg = EnvConfig(cost=cost, episode_days=episode_days)
    env = InventoryEnvironment(env_cfg)
    env.attach_series(
        forecast_point=series["forecast_point"],
        forecast_std=series["forecast_std"],
        actual_demand=series["actual_demand"],
        lead_time_mean=series["lead_time_mean"],
        lead_time_std=series["lead_time_std"],
        supplier_risk=series["supplier_risk"],
        initial_inventory=float(np.mean(series["forecast_point"][:30]) * 30.0),
        seed=seed,
    )
    state = env.reset(start_index=start_index)

    annual_demand = float(np.mean(series["forecast_point"]) * 365.0)
    critical_ratio = (
        cost.shortage_cost_per_unit
        / max(cost.shortage_cost_per_unit + cost.unit_cost * cost.holding_rate_annual, 1e-6)
    )

    done = False
    while not done:
        expected_ltd = float(env.forecast_point[env.t] * env.lead_time_mean[env.t])
        if name == "never_order":
            a = never_order(env, state, expected_ltd)
        elif name == "newsvendor":
            a = newsvendor(env, state, expected_ltd, critical_ratio=critical_ratio)
        elif name == "eoq":
            a = economic_order_quantity(env, state, expected_ltd, annual_demand=annual_demand)
        elif name == "sS":
            a = sS_policy(env, state, expected_ltd)
        elif name == "RQ":
            a = RQ_policy(env, state, expected_ltd)
        else:
            raise ValueError(f"Unknown baseline: {name}")
        state, _r, done, _ = env.step(a)

    summary = env.episode_summary()
    return BaselineResult(
        name=name,
        total_cost=summary["total_cost"],
        service_level=summary["service_level"],
        unmet_units=summary["unmet_units"],
        n_orders=summary["n_orders"],
        avg_inventory=summary["avg_inventory"],
        reward=summary["reward"],
    )


ALL_BASELINES = ("never_order", "newsvendor", "eoq", "sS", "RQ")