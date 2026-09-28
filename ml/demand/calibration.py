"""Interval calibration diagnostics.

Given a set of (lower, upper, actual) triplets, compute empirical coverage
for a nominal confidence level. A well-calibrated model has:

    empirical_coverage ≈ nominal_coverage

Two diagnostics are returned:
- per-horizon coverage
- coverage reliability diagram (nominal vs empirical) for a set of levels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class CoverageReport:
    nominal: float
    empirical_overall: float
    per_horizon: List[float] = field(default_factory=list)
    n: int = 0

    def to_dict(self) -> dict:
        return {
            "nominal": self.nominal,
            "empirical_overall": self.empirical_overall,
            "per_horizon": self.per_horizon,
            "n": self.n,
        }


def coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
    *,
    nominal: float = 0.8,
) -> CoverageReport:
    """`lower`, `upper`, `actual` are all (N, H)."""
    mask = (actual >= lower) & (actual <= upper)
    overall = float(np.mean(mask))
    per_h = [float(np.mean(mask[:, h])) for h in range(mask.shape[1])]
    return CoverageReport(
        nominal=nominal,
        empirical_overall=overall,
        per_horizon=per_h,
        n=int(mask.size),
    )


def reliability_curve(
    quantile_preds: Dict[str, np.ndarray],
    actual: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Return {q: {nominal, empirical}} for every quantile key 'q<X>' in the dict."""
    out: Dict[str, Dict[str, float]] = {}
    for key, preds in quantile_preds.items():
        try:
            q = float(key.lstrip("q"))
        except ValueError:
            continue
        # For a lower-side quantile, empirical coverage = P(actual <= pred).
        empirical = float(np.mean(actual <= preds))
        out[key] = {"nominal": q, "empirical": empirical}
    return out