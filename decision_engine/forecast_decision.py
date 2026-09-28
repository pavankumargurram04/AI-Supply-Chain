"""Demand forecast decision step.

Calls the demand predictor and applies any scenario multiplier. Returns a
plain dict that becomes part of the DecisionTrace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.logging import get_logger
from decision_engine.state_builder import DecisionState

log = get_logger(__name__)


@dataclass
class ForecastResult:
    point: np.ndarray
    std: np.ndarray
    dates: List[str]
    origin_date: str
    model_version: str
    feature_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point": [float(x) for x in self.point],
            "std": [float(x) for x in self.std],
            "dates": list(self.dates),
            "origin_date": self.origin_date,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "horizon": int(self.point.size),
            "mean": float(np.mean(self.point)),
            "sum": float(np.sum(self.point)),
        }


def produce_forecast(
    state: DecisionState,
    *,
    demand_checkpoint_dir: str | Path,
    device: str = "cpu",
) -> ForecastResult:
    from ml.demand.predict import DemandPredictor

    predictor = DemandPredictor(demand_checkpoint_dir, device=device)
    fc = predictor.predict(
        state.feature_frame,
        product_sku=state.product_sku,
        warehouse_code=state.warehouse_code,
    )

    point = np.asarray(fc.point, dtype=np.float64) * float(state.demand_multiplier)

    # Uncertainty: prefer the quantile head; else derive from MC-dropout.
    if fc.quantiles and "q0.1" in fc.quantiles and "q0.9" in fc.quantiles:
        lo = np.asarray(fc.quantiles["q0.1"])
        hi = np.asarray(fc.quantiles["q0.9"])
        std = (hi - lo) / 2.563 * float(state.demand_multiplier)
    elif fc.lower is not None and fc.upper is not None:
        lo = np.asarray(fc.lower)
        hi = np.asarray(fc.upper)
        std = (hi - lo) / 2.563 * float(state.demand_multiplier)
    else:
        # Fallback: 20 % of the point forecast.
        std = 0.20 * point

    return ForecastResult(
        point=point,
        std=std,
        dates=list(fc.forecast_dates),
        origin_date=fc.origin_date,
        model_version=fc.model_version,
        feature_version=fc.feature_version,
    )