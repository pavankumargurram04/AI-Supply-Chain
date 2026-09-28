"""Deterministic inference wrapper for the DQN policy.

Loads a checkpoint and given a state vector (or a full series) returns the
greedy action index and the corresponding order quantity in units.

Same input + same checkpoint → same output. No sampling, no dropout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ml.inventory.cost_model import CostParameters
from ml.inventory.dqn_model import DQNAgent
from ml.inventory.environment import EnvConfig, InventoryEnvironment


@dataclass
class PolicyDecision:
    action_index: int
    multiplier: float
    order_quantity: float
    expected_lead_time_demand: float
    state: List[float]
    model_version: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action_index": int(self.action_index),
            "multiplier": float(self.multiplier),
            "order_quantity": float(self.order_quantity),
            "expected_lead_time_demand": float(self.expected_lead_time_demand),
            "state": list(self.state),
            "model_version": self.model_version,
            "extras": self.extras,
        }


class InventoryPolicy:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cpu") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        meta_path = self.checkpoint_dir / "training_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata: {meta_path}")
        self.meta = json.loads(meta_path.read_text())

        self.action_multipliers = list(self.meta["action_multipliers"])
        self.env_cfg = self.meta["env_config"]
        self.device = torch.device(device)

        ckpt_path = self.checkpoint_dir / "dqn_best.pt"
        if not ckpt_path.exists():
            ckpt_path = self.checkpoint_dir / "dqn_last.pt"
        self.agent = DQNAgent.load(str(ckpt_path), device=self.device)
        self.agent.eval()

        self.model_version = self.meta.get("model_version", "dqn_inventory_unnamed")

    @torch.no_grad()
    def decide_from_state(
        self,
        state: np.ndarray,
        *,
        expected_lead_time_demand: float,
    ) -> PolicyDecision:
        q = self.agent.q_values(torch.from_numpy(state[None, :].astype(np.float32)).to(self.device))
        action = int(q.argmax(dim=1).item())
        mult = float(self.action_multipliers[action])
        order_qty = mult * float(expected_lead_time_demand)
        return PolicyDecision(
            action_index=action,
            multiplier=mult,
            order_quantity=order_qty,
            expected_lead_time_demand=float(expected_lead_time_demand),
            state=[float(x) for x in state],
            model_version=self.model_version,
        )

    def decide_for_series_step(
        self,
        *,
        forecast_point: np.ndarray,
        forecast_std: np.ndarray,
        actual_demand: np.ndarray,
        lead_time_mean: np.ndarray,
        lead_time_std: np.ndarray,
        supplier_risk: np.ndarray,
        initial_inventory: float,
        step_index: int,
        service_target: float = 0.95,
        unit_cost: float = 10.0,
    ) -> PolicyDecision:
        """Run the environment up to `step_index` with greedy actions, return the
        decision at that step. Useful for demoing the policy on a specific day.
        """
        cost = CostParameters(
            unit_cost=unit_cost,
            holding_rate_annual=self.env_cfg["cost"]["holding_rate_annual"],
            order_fixed_cost=self.env_cfg["cost"]["order_fixed_cost"],
            shortage_cost_per_unit=self.env_cfg["cost"]["shortage_cost_per_unit"],
            shipping_cost_per_order=self.env_cfg["cost"]["shipping_cost_per_order"],
        )
        env_cfg = EnvConfig(cost=cost, episode_days=step_index + 1)
        env = InventoryEnvironment(env_cfg)
        env.attach_series(
            forecast_point=forecast_point,
            forecast_std=forecast_std,
            actual_demand=actual_demand,
            lead_time_mean=lead_time_mean,
            lead_time_std=lead_time_std,
            supplier_risk=supplier_risk,
            initial_inventory=initial_inventory,
            service_target=service_target,
            seed=0,
        )
        state = env.reset(start_index=0)
        last_decision = None
        for _ in range(step_index + 1):
            expected_ltd = float(env.forecast_point[env.t] * env.lead_time_mean[env.t])
            decision = self.decide_from_state(state, expected_lead_time_demand=expected_ltd)
            last_decision = decision
            state, _r, done, _ = env.step(decision.action_index)
            if done:
                break
        if last_decision is None:
            raise RuntimeError("No decision produced.")
        return last_decision