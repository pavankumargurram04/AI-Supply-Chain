"""Distribution drift metrics: PSI and KS.

- PSI (Population Stability Index): binned histogram comparison. Common
  interpretation thresholds (documented, not used as hidden decisions):
      PSI < 0.1   — no significant shift
      PSI < 0.25  — moderate shift (alert at WARNING)
      PSI >= 0.25 — significant shift (alert at CRITICAL)

- KS (Kolmogorov–Smirnov): two-sample statistic. We report the raw
  statistic and a p-value; the p-value is not used as a business rule.

Both metrics operate on 1-D numeric arrays. No categorical support yet —
categorical drift is handled by `data_quality.schema_drift`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from scipy.stats import ks_2samp  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------
def psi(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index between two 1-D samples.

    Bins are derived from the reference distribution quantiles. For
    degenerate or near-degenerate reference distributions, fixed-width
    bins are used so that genuine shifts are not hidden by collapsed
    quantile edges.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if ref.size == 0 or cur.size == 0:
        return 0.0

    ref_min = float(np.min(ref))
    ref_max = float(np.max(ref))
    ref_center = float(np.median(ref))

    # Detect a genuinely constant reference distribution.
    ref_scale = max(1.0, abs(ref_center))
    tolerance = 1e-6 * ref_scale

    if (ref_max - ref_min) <= tolerance:
        cur_min = float(np.min(cur))
        cur_max = float(np.max(cur))

        # Same effectively-constant distribution -> no drift.
        if (
            (cur_max - cur_min) <= tolerance
            and abs(float(np.median(cur)) - ref_center) <= tolerance
        ):
            return 0.0

        # Build fixed bins around the reference point so a shifted
        # distribution cannot disappear because quantile edges collapse.
        delta = max(
            tolerance,
            abs(ref_center) * 0.01,
            float(np.std(cur)) * 0.01,
        )
        edges = np.array(
            [-np.inf, ref_center - delta, ref_center + delta, np.inf],
            dtype=float,
        )
    else:
        # Reference-derived equal-mass bins.
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        edges = np.quantile(ref, qs)

        # Remove duplicate quantile edges caused by repeated values.
        edges = np.unique(edges)

        if edges.size == 0:
            delta = max(
                tolerance,
                abs(ref_center) * 0.01,
                float(np.std(ref)) * 0.01,
            )
            edges = np.array(
                [ref_center - delta, ref_center + delta],
                dtype=float,
            )

        edges = np.concatenate([[-np.inf], edges, [np.inf]])

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    n_ref_bins = len(ref_counts)

    ref_p = (ref_counts + eps) / (
        ref_counts.sum() + eps * n_ref_bins
    )
    cur_p = (cur_counts + eps) / (
        cur_counts.sum() + eps * n_ref_bins
    )

    return float(
        np.sum((cur_p - ref_p) * np.log(cur_p / ref_p))
    )


def psi_severity(value: float) -> str:
    """Documented thresholds — for alert severity only, not decisions."""
    if value < 0.10:
        return "info"
    if value < 0.25:
        return "warning"
    return "critical"


# ---------------------------------------------------------------------------
# KS
# ---------------------------------------------------------------------------
def ks_statistic(reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Return KS statistic + approximate p-value. Uses scipy if available."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    if _HAS_SCIPY:
        res = ks_2samp(ref, cur)
        return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}

    # Manual two-sample KS statistic (no p-value).
    all_vals = np.concatenate([ref, cur])
    all_vals.sort()
    cdf_ref = np.searchsorted(np.sort(ref), all_vals, side="right") / ref.size
    cdf_cur = np.searchsorted(np.sort(cur), all_vals, side="right") / cur.size
    stat = float(np.max(np.abs(cdf_ref - cdf_cur)))
    return {"statistic": stat, "p_value": float("nan")}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    n_reference: int
    n_current: int

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": float(self.psi),
            "ks_statistic": float(self.ks_statistic),
            "ks_p_value": float(self.ks_p_value),
            "n_reference": int(self.n_reference),
            "n_current": int(self.n_current),
            "psi_severity": psi_severity(self.psi),
        }


def drift_report(
    reference_df,
    current_df,
    *,
    features: Optional[List[str]] = None,
    n_bins: int = 10,
) -> List[FeatureDrift]:
    """Compute PSI + KS for each numeric feature.

    `reference_df` / `current_df` are pandas DataFrames. If `features` is
    None, every shared numeric column is used.
    """
    if reference_df is None or current_df is None:
        return []

    common = set(reference_df.columns) & set(current_df.columns)
    if features is None:
        candidate = [
            c for c in common
            if np.issubdtype(reference_df[c].dtype, np.number)
            and np.issubdtype(current_df[c].dtype, np.number)
        ]
    else:
        candidate = [c for c in features if c in common]

    out: List[FeatureDrift] = []
    for feat in sorted(candidate):
        ref = reference_df[feat].to_numpy(dtype=float)
        cur = current_df[feat].to_numpy(dtype=float)
        p = psi(ref, cur, n_bins=n_bins)
        ks = ks_statistic(ref, cur)
        out.append(FeatureDrift(
            feature=feat, psi=p,
            ks_statistic=ks["statistic"], ks_p_value=ks["p_value"],
            n_reference=int(np.isfinite(ref).sum()),
            n_current=int(np.isfinite(cur).sum()),
        ))
    return out
