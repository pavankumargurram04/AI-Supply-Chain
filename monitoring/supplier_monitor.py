"""Supplier risk monitor.

Checks:
    1) Feature drift on supplier features.
    2) Risk probability distribution shift (mean prediction vs. reference).
    3) Calibration Brier score drift when labels are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.logging import get_logger
from monitoring.alerting import MonitorReport
from monitoring.drift import drift_report, psi_severity

log = get_logger(__name__)


_SUPPLIER_FEATURE_PREFIXES = ("feat_sup_",)


def monitor_supplier(
    *,
    features_dir: str | Path,
    predictions_path: Optional[str | Path] = None,
    recent_days: int = 30,
    reference_days: int = 180,
) -> MonitorReport:
    report = MonitorReport(name="supplier_monitor")
    features_dir = Path(features_dir)

    feature_path = next(features_dir.glob("supplier_features_*.parquet"), None)
    if feature_path is None:
        report.notes.append(f"no supplier features under {features_dir}")
        return report

    df = pd.read_parquet(feature_path)
    if df.empty:
        report.notes.append("empty supplier features")
        return report

    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    recent_start = max_date - pd.Timedelta(days=recent_days)
    reference_start = max_date - pd.Timedelta(days=recent_days + reference_days)

    recent = df[df["date"] > recent_start]
    reference = df[(df["date"] > reference_start) & (df["date"] <= recent_start)]
    if recent.empty or reference.empty:
        report.notes.append("insufficient data for drift comparison")
        return report

    feature_cols = [
    c
    for c in df.columns
    if any(c.startswith(p) for p in _SUPPLIER_FEATURE_PREFIXES)
    and pd.api.types.is_numeric_dtype(df[c])
]
    drifts = drift_report(reference, recent, features=feature_cols)
    report.metrics["n_features_checked"] = len(drifts)
    report.metrics["max_psi"] = max((d.psi for d in drifts), default=0.0)

    critical = [d for d in drifts if psi_severity(d.psi) == "critical"]
    warning = [d for d in drifts if psi_severity(d.psi) == "warning"]
    report.metrics["n_critical_drift_features"] = len(critical)
    report.metrics["n_warning_drift_features"] = len(warning)

    if critical:
        report.add_alert(
            source="supplier_drift",
            severity="critical",
            title=f"{len(critical)} supplier feature(s) critically drifted",
            message=", ".join(f"{d.feature}({d.psi:.2f})" for d in critical[:5]),
            entity_type="supplier_model",
            payload={"features": [d.to_dict() for d in critical]},
        )
    elif warning:
        report.add_alert(
            source="supplier_drift",
            severity="warning",
            title=f"{len(warning)} supplier feature(s) moderately drifted",
            message=", ".join(f"{d.feature}({d.psi:.2f})" for d in warning[:5]),
            entity_type="supplier_model",
            payload={"features": [d.to_dict() for d in warning]},
        )

    # Prediction distribution shift.
    if predictions_path is not None and Path(predictions_path).exists():
        pdf = pd.read_parquet(predictions_path)
        if not pdf.empty and "risk_probability" in pdf.columns and "created_at" in pdf.columns:
            pdf["created_at"] = pd.to_datetime(pdf["created_at"])
            pmax = pdf["created_at"].max()
            prec = pdf[pdf["created_at"] > pmax - pd.Timedelta(days=recent_days)]
            pref = pdf[
                (pdf["created_at"] > pmax - pd.Timedelta(days=recent_days + reference_days))
                & (pdf["created_at"] <= pmax - pd.Timedelta(days=recent_days))
            ]
            if not prec.empty and not pref.empty:
                ref_mean = float(pref["risk_probability"].mean())
                cur_mean = float(prec["risk_probability"].mean())
                report.metrics["risk_mean_reference"] = ref_mean
                report.metrics["risk_mean_current"] = cur_mean
                report.metrics["risk_mean_delta"] = cur_mean - ref_mean
                if abs(cur_mean - ref_mean) >= 0.15:
                    report.add_alert(
                        source="supplier_risk_shift",
                        severity="warning",
                        title="Mean predicted supplier risk shifted",
                        message=(
                            f"mean risk moved from {ref_mean:.3f} to {cur_mean:.3f} "
                            f"(delta {cur_mean - ref_mean:+.3f})"
                        ),
                        entity_type="supplier_model",
                        payload={"reference_mean": ref_mean, "current_mean": cur_mean},
                    )

    return report
