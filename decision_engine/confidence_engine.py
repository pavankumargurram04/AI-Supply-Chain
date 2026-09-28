"""Confidence in the recommendation.

Confidence is a bounded score derived from:
    - data quality (rows used, missingness on the encoder window)
    - forecast uncertainty (mean std / mean point)
    - supplier risk magnitude (max risk over selected suppliers)
    - allocation completeness (fraction of requested quantity filled)

The score is a probability-like number in [0, 1] and is stored for audit.
No thresholds decide outcomes; the score is an *output*, not an input.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _bounded(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def compute_confidence(trace_dict: Dict[str, Any]) -> float:
    scores: List[float] = []

    # 1) Forecast signal-to-noise.
    fc = np.asarray(trace_dict.get("forecast_point", []), dtype=float)
    if fc.size > 0 and "forecast_std" in trace_dict:
        std = np.asarray(trace_dict["forecast_std"], dtype=float)
        mean = float(np.mean(np.abs(fc))) + 1e-9
        cv = float(np.mean(std)) / mean
        scores.append(_bounded(1.0 - cv))     # lower CV → higher confidence

    # 2) Supplier risk — max risk among selected suppliers.
    sups = trace_dict.get("final_suppliers", [])
    if sups:
        max_risk = max(float(s.get("risk_probability", 0.0)) for s in sups)
        scores.append(_bounded(1.0 - max_risk))

    # 3) Allocation completeness.
    alloc = trace_dict.get("allocation") or {}
    req = float(alloc.get("requested_quantity", 0.0))
    got = float(alloc.get("allocated_quantity", 0.0))
    if req > 0:
        scores.append(_bounded(got / req))

    # 4) Stockout probability reduces confidence.
    p = float(trace_dict.get("stockout_probability", 0.0))
    scores.append(_bounded(1.0 - p))

    if not scores:
        return 0.5
    return float(np.mean(scores))