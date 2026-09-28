"""Demand agent.

Wraps `decision_engine.forecast_decision.produce_forecast`. The payload
contains the forecast point, uncertainty, dates, and provenance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agents.base_agent import BaseAgent
from core.logging import get_logger
from decision_engine import forecast_decision

log = get_logger(__name__)


class DemandAgent(BaseAgent):
    name = "demand"

    def run(self, context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
        state = context["state"]
        checkpoint_dir = context["demand_checkpoint_dir"]
        device = context.get("device", "cpu")

        forecast = forecast_decision.produce_forecast(
            state, demand_checkpoint_dir=checkpoint_dir, device=device,
        )

        payload = forecast.to_dict()
        model_versions = {"demand_model": forecast.model_version}
        warnings: List[str] = []
        # Surface low mean forecast as a warning (not a rule, just a signal).
        if payload.get("mean", 0.0) <= 0.0:
            warnings.append("mean forecast is zero or negative")
        return payload, model_versions, warnings