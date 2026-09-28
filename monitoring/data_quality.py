"""Data quality monitors.

Three independent checks:
    - missingness drift (fraction missing per feature vs. a reference window)
    - schema drift (feature set differs from reference)
    - range drift (min/max outside the reference range)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    n_rows_reference: int
    n_rows_current: int
    missingness_drift: Dict[str, Dict[str, float]] = field(default_factory=dict)
    schema_added: List[str] = field(default_factory=list)
    schema_removed: List[str] = field(default_factory=list)
    range_violations: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_rows_reference": self.n_rows_reference,
            "n_rows_current": self.n_rows_current,
            "missingness_drift": self.missingness_drift,
            "schema_added": list(self.schema_added),
            "schema_removed": list(self.schema_removed),
            "range_violations": self.range_violations,
        }


def check_data_quality(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    missingness_alert_delta: float = 0.05,
    range_alert_fraction: float = 0.01,
) -> DataQualityReport:
    """Compare a reference window to a current window.

    `missingness_alert_delta` — alert when the missing rate increases by more
    than this fraction.
    `range_alert_fraction` — alert when more than this fraction of current
    rows fall outside the reference min/max.
    """
    report = DataQualityReport(
        n_rows_reference=int(len(reference_df)),
        n_rows_current=int(len(current_df)),
    )
    if reference_df is None or current_df is None:
        return report

    ref_cols = set(reference_df.columns)
    cur_cols = set(current_df.columns)
    report.schema_added = sorted(cur_cols - ref_cols)
    report.schema_removed = sorted(ref_cols - cur_cols)

    common = sorted(ref_cols & cur_cols)
    for col in common:
        ref_miss = float(reference_df[col].isna().mean())
        cur_miss = float(current_df[col].isna().mean())
        delta = cur_miss - ref_miss
        if abs(delta) >= missingness_alert_delta:
            report.missingness_drift[col] = {
                "reference": ref_miss,
                "current": cur_miss,
                "delta": delta,
            }

    for col in common:
        if not np.issubdtype(reference_df[col].dtype, np.number):
            continue
        ref = reference_df[col].dropna().to_numpy(dtype=float)
        cur = current_df[col].dropna().to_numpy(dtype=float)
        if ref.size == 0 or cur.size == 0:
            continue
        lo, hi = float(np.min(ref)), float(np.max(ref))
        outside = np.mean((cur < lo) | (cur > hi))
        if outside >= range_alert_fraction:
            report.range_violations[col] = {
                "reference_min": lo, "reference_max": hi,
                "fraction_outside": float(outside),
            }

    return report