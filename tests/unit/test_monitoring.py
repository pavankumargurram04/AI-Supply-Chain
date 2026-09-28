"""Monitoring tests.

Covers PSI/KS drift, data quality, alerting, and each monitor's behaviour
on tiny synthetic inputs.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# PSI / KS (pure)
# ---------------------------------------------------------------------------
def test_psi_zero_on_identical_samples():
    from monitoring.drift import psi

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=1000)
    assert psi(a, a) < 0.02


def test_psi_large_on_shifted_samples():
    from monitoring.drift import psi

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=1000)
    b = rng.normal(3, 1, size=1000)
    assert psi(a, b) > 0.25


def test_psi_severity_thresholds():
    from monitoring.drift import psi_severity

    assert psi_severity(0.05) == "info"
    assert psi_severity(0.15) == "warning"
    assert psi_severity(0.40) == "critical"


def test_ks_statistic_range_and_direction():
    from monitoring.drift import ks_statistic

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    b = rng.normal(3, 1, 500)
    res = ks_statistic(a, b)
    assert 0.0 <= res["statistic"] <= 1.0
    assert res["statistic"] > 0.5


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
def test_data_quality_detects_missingness_and_schema_drift():
    from monitoring.data_quality import check_data_quality

    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    cur = pd.DataFrame({
        "a": [1.0, None, None],       # 66% missing
        "c": [1.0, 2.0, 3.0],         # new column
    })
    rep = check_data_quality(ref, cur, missingness_alert_delta=0.05)
    assert "a" in rep.missingness_drift
    assert "c" in rep.schema_added
    assert "b" in rep.schema_removed


def test_data_quality_detects_range_violation():
    from monitoring.data_quality import check_data_quality

    ref = pd.DataFrame({"a": list(np.linspace(0, 1, 100))})
    cur = pd.DataFrame({"a": list(np.linspace(5, 6, 100))})
    rep = check_data_quality(ref, cur, range_alert_fraction=0.01)
    assert "a" in rep.range_violations


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------
def test_monitor_report_dedupes_within_call():
    from monitoring.alerting import MonitorReport

    rep = MonitorReport(name="test")
    for _ in range(3):
        rep.add_alert(
            source="x", severity="warning", title="dup",
            message="same", entity_type="e", entity_id=1,
        )
    keys = {a.dedupe_key() for a in rep.alerts}
    assert len(keys) == 1


# ---------------------------------------------------------------------------
# Demand monitor (filesystem only)
# ---------------------------------------------------------------------------
def _make_demand_features(tmp_path: Path) -> Path:
    """Write 300 days of demand features with an obvious drift in the last 30."""
    from data_platform.features.feature_registry import FEATURE_VERSION

    rows = []
    start = date(2023, 1, 1)
    for d in range(300):
        dt = start + timedelta(days=d)
        # Drift: after day 270 the mean sharply increases.
        base = 10.0 if d < 270 else 100.0
        rows.append({
            "product_sku": "SKU-000", "warehouse_code": "WH-A",
            "date": pd.Timestamp(dt), "quantity": base,
            "feat_demand_lag_1": base,
            "feat_demand_roll_mean_7": base,
            "feat_demand_roll_std_28": 1.0,
            "feat_demand_expand_mean": base,
            "feat_promo_count_7": 0, "feat_holiday_count_7": 0,
            "feat_censored_count_7": 0,
        })
    fdir = tmp_path / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        fdir / f"demand_features_{FEATURE_VERSION}.parquet", index=False,
    )
    return fdir


def test_demand_monitor_detects_drift(tmp_path):
    from monitoring.forecast_monitor import monitor_demand

    fdir = _make_demand_features(tmp_path)
    report = monitor_demand(
        features_dir=fdir,
        recent_days=30, reference_days=180,
    )
    # Because the mean feature jumps by 10× in the recent window, PSI must be
    # flagged at least as a warning.
    assert report.metrics["max_psi"] > 0.10
    assert any(a.source == "forecast_drift" for a in report.alerts)


def test_demand_monitor_no_drift_on_stable_series(tmp_path):
    from monitoring.forecast_monitor import monitor_demand
    from data_platform.features.feature_registry import FEATURE_VERSION

    rows = []
    start = date(2023, 1, 1)
    rng = np.random.default_rng(0)
    for d in range(300):
        dt = start + timedelta(days=d)
        base = 10.0 + rng.normal(0, 0.1)
        rows.append({
            "product_sku": "SKU-000", "warehouse_code": "WH-A",
            "date": pd.Timestamp(dt), "quantity": base,
            "feat_demand_lag_1": base, "feat_demand_roll_mean_7": base,
            "feat_demand_roll_std_28": 0.1, "feat_demand_expand_mean": base,
            "feat_promo_count_7": 0, "feat_holiday_count_7": 0,
            "feat_censored_count_7": 0,
        })
    fdir = tmp_path / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        fdir / f"demand_features_{FEATURE_VERSION}.parquet", index=False,
    )

    report = monitor_demand(features_dir=fdir, recent_days=30, reference_days=180)
    assert report.metrics["max_psi"] < 0.10


# ---------------------------------------------------------------------------
# Inventory monitor
# ---------------------------------------------------------------------------
def test_inventory_monitor_flags_high_stockouts(tmp_path):
    from monitoring.inventory_monitor import monitor_inventory

    rows = []
    start = date(2023, 1, 1)
    for d in range(200):
        dt = start + timedelta(days=d)
        mv = "stockout" if d > 150 else "issue"
        rows.append({
            "product_sku": "SKU-000", "warehouse_code": "WH-A",
            "movement_type": mv, "quantity": -1.0,
            "occurred_at": pd.Timestamp(dt),
        })
    p = tmp_path / "inventory_movements"
    p.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(p / "data.parquet", index=False)

    report = monitor_inventory(
        movements_path=p / "data.parquet",
        recent_days=30, reference_days=120,
        stockout_alert_threshold=10,
    )
    assert any(a.source == "inventory_stockouts" for a in report.alerts)


# ---------------------------------------------------------------------------
# Performance trend
# ---------------------------------------------------------------------------
def test_rolling_performance_and_trend():
    from monitoring.performance import rolling_performance, trend

    # 180 days of residuals with a slowly increasing error.
    rows = []
    start = date(2023, 1, 1)
    for d in range(180):
        dt = start + timedelta(days=d)
        err = 1.0 + (d / 180.0) * 5.0
        rows.append({"origin_date": pd.Timestamp(dt),
                     "y_pred": 10.0 + err, "y_true": 10.0})
    df = pd.DataFrame(rows)
    windows = rolling_performance(df, date_col="origin_date",
                                  pred_col="y_pred", target_col="y_true",
                                  freq="30D")
    assert len(windows) >= 4
    tr = trend(windows, metric="wape")
    assert tr is not None
    assert tr["delta"] > 0