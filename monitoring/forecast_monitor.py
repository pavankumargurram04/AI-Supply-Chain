"""Demand forecast monitor.

Runs two checks:
    1) Feature drift: compare the latest 30 days of demand features to a
       reference window (default: the previous 180 days).
    2) Residual performance: bucket residuals by 30-day windows and flag a
       significant trend in WAPE or bias.

Writes alerts to the report. Caller persists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.logging import get_logger
from monitoring.alerting import MonitorReport
from monitoring.drift import drift_report, psi_severity
from monitoring.performance import rolling_performance, trend

log = get_logger(__name__)


_DEMAND_FEATURE_PREFIXES = (
    "feat_demand_", "feat_promo_", "feat_holiday_", "feat_censored_",
)


def monitor_demand(
    *,
    features_dir: str | Path,
    residuals_path: Optional[str | Path] = None,
    recent_days: int = 30,
    reference_days: int = 180,
    psi_warning: float = 0.10,
    psi_critical: float = 0.25,
) -> MonitorReport:
    """Return a MonitorReport for the demand model."""
    report = MonitorReport(name="demand_monitor")
    features_dir = Path(features_dir)

    feature_path = next(features_dir.glob("demand_features_*.parquet"), None)
    if feature_path is None:
        report.notes.append(f"no demand features under {features_dir}")
        return report

    df = pd.read_parquet(feature_path)
    if df.empty:
        report.notes.append("empty demand features")
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
        c for c in df.columns
        if any(c.startswith(p) for p in _DEMAND_FEATURE_PREFIXES)
    ]
    drifts = drift_report(reference, recent, features=feature_cols, n_bins=3)
    report.metrics["n_features_checked"] = len(drifts)
    report.metrics["n_recent_rows"] = int(len(recent))
    report.metrics["n_reference_rows"] = int(len(reference))

    critical_drifts = []
    warning_drifts = []
    for d in drifts:
        sev = psi_severity(d.psi)
        if sev == "critical":
            critical_drifts.append(d)
        elif sev == "warning":
            warning_drifts.append(d)

    report.metrics["n_critical_drift_features"] = len(critical_drifts)
    report.metrics["n_warning_drift_features"] = len(warning_drifts)
    report.metrics["max_psi"] = max((d.psi for d in drifts), default=0.0)

    if critical_drifts:
        report.add_alert(
            source="forecast_drift",
            severity="critical",
            title=f"{len(critical_drifts)} demand feature(s) critically drifted",
            message=(
                f"PSI ≥ {psi_critical} on: "
                + ", ".join(f"{d.feature}({d.psi:.2f})" for d in critical_drifts[:5])
            ),
            entity_type="demand_model",
            payload={"features": [d.to_dict() for d in critical_drifts]},
        )
    elif warning_drifts:
        report.add_alert(
            source="forecast_drift",
            severity="warning",
            title=f"{len(warning_drifts)} demand feature(s) moderately drifted",
            message=(
                f"PSI in [{psi_warning},{psi_critical}) on: "
                + ", ".join(f"{d.feature}({d.psi:.2f})" for d in warning_drifts[:5])
            ),
            entity_type="demand_model",
            payload={"features": [d.to_dict() for d in warning_drifts]},
        )

    # Residual performance.
    if residuals_path is None:
        default_path = Path("datasets/curated/residuals_demand.parquet")
        residuals_path = default_path if default_path.exists() else None

    if residuals_path and Path(residuals_path).exists():
        rdf = pd.read_parquet(residuals_path)
        if not rdf.empty:
            perf = rolling_performance(
                rdf, date_col="origin_date",
                pred_col="y_pred", target_col="y_true", freq="30D",
            )
            if perf:
                tr = trend(perf, metric="wape")
                report.metrics["residual_windows"] = [w.to_dict() for w in perf[-6:]]
                if tr is not None:
                    report.metrics["wape_trend"] = tr
                    if tr["delta"] > 0.05:
                        report.add_alert(
                            source="forecast_performance",
                            severity="warning",
                            title="WAPE has degraded over recent windows",
                            message=(
                                f"WAPE moved from {tr['first']:.3f} to {tr['last']:.3f} "
                                f"(delta {tr['delta']:+.3f})"
                            ),
                            entity_type="demand_model",
                            payload=tr,
                        )
    else:
        report.notes.append("residuals unavailable; performance check skipped")

    return report

