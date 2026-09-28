"""Inventory agent.

Wraps `decision_engine.inventory_decision.run_inventory_decision`. Consumes
the demand agent's output directly (forecast) as well as the state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from agents.base_agent import BaseAgent
from core.logging import get_logger
from decision_engine.forecast_decision import ForecastResult
from decision_engine.inventory_decision import run_inventory_decision

log = get_logger(__name__)


class InventoryAgent(BaseAgent):
    name = "inventory"

    def run(self, context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
        state = context["state"]
        demand_msg = context.get("demand_message")
        if demand_msg is None or demand_msg.status == "error":
            raise RuntimeError("Inventory agent requires a successful demand message.")

        # Rebuild the ForecastResult from the demand agent payload.
        payload = demand_msg.payload
        forecast = ForecastResult(
            point=np.asarray(payload["point"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            dates=list(payload["dates"]),
            origin_date=str(payload["origin_date"]),
            model_version=str(payload["model_version"]),
            feature_version=str(payload["feature_version"]),
        )

        simulation_samples = context.get("simulation_samples", 10_000)
        decision = run_inventory_decision(
            state, forecast, simulation_samples=simulation_samples,
        )

        out_payload = decision.to_dict()
        model_versions = {"demand_model": forecast.model_version}
        warnings: List[str] = []
        if decision.stockout_probability >= 0.5:
            warnings.append(
                f"high stockout probability: {decision.stockout_probability:.2%}"
            )
        if decision.mode == "gaussian_fallback":
            warnings.append("LTD simulation used Gaussian fallback (no residuals available)")
        return out_payload, model_versions, warnings