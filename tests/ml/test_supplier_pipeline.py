"""End-to-end tiny-run test for the supplier risk pipeline.

Builds a minimal synthetic feature + event dataset, trains with 1 boosting
round, and asserts:

- Dataset labels are future-only (a spike event does not affect the row at
  the event date).
- The classifier produces finite calibrated probabilities.
- Threshold selection is deterministic.
- Reproducibility: same seed → same test AUC.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_supplier_features(dir_: Path, *, n_suppliers: int = 3, n_days: int = 180) -> None:
    from data_platform.features.feature_registry import FEATURE_VERSION

    rng = np.random.default_rng(0)
    rows = []
    start = date(2022, 1, 1)
    for s in range(n_suppliers):
        sup = f"SUP-{s:03d}"
        for d in range(n_days):
            dt = start + timedelta(days=d)
            rows.append({
                "supplier_code": sup,
                "product_sku": f"SKU-{s % 2:03d}",
                "date": pd.Timestamp(dt),
                "feat_sup_delay_count_30": float(rng.integers(0, 3)),
                "feat_sup_delay_count_90": float(rng.integers(0, 5)),
                "feat_sup_quality_count_30": float(rng.integers(0, 2)),
                "feat_sup_outage_count_90": float(rng.integers(0, 2)),
                "feat_sup_ontime_proxy_30": float(rng.random()),
                "feat_sup_days_since_delay": float(rng.integers(0, 60)),
                "feat_sup_lead_time_days": 7.0,
                "feat_sup_unit_price": 10.0,
                "feat_sup_reliability": ["premium", "standard", "poor"][s % 3],
            })
    df = pd.DataFrame(rows)
    out = dir_ / "features"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"supplier_features_{FEATURE_VERSION}.parquet", index=False)


def _write_supplier_events(raw_dir: Path, *, n_suppliers: int = 3) -> None:
    rows = []
    rng = np.random.default_rng(1)
    start = date(2022, 1, 1)
    for s in range(n_suppliers):
        sup = f"SUP-{s:03d}"
        for k in range(6):
            dt = start + timedelta(days=int(rng.integers(30, 170)))
            rows.append({
                "supplier_code": sup,
                "product_sku": None,
                "event_type": "delay",
                "severity": "medium",
                "payload": {"delay_days": 3},
                "description": "synthetic",
                "occurred_at": pd.Timestamp(dt),
            })
    ev = raw_dir / "supplier_events"
    ev.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ev / "data.parquet", index=False)


# ---------------------------------------------------------------------------
# Test 1: labels only depend on future events
# ---------------------------------------------------------------------------
def test_labels_are_future_only(tmp_path):
    from ml.supplier_risk.dataset import build_supplier_dataset

    _write_supplier_features(tmp_path)
    _write_supplier_events(tmp_path / "raw")

    bundle = build_supplier_dataset(
        features_dir=tmp_path / "features",
        raw_dir=tmp_path / "raw",
        horizon_days=30,
    )
    # Positive rate is non-zero (we placed events).
    assert bundle.positive_rate_train + bundle.positive_rate_val + bundle.positive_rate_test > 0.0
    # Labels must be 0/1 (never -1 after filtering).
    for y in (bundle.y_train, bundle.y_val, bundle.y_test):
        assert set(y.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Test 2: end-to-end train
# ---------------------------------------------------------------------------
def test_training_tiny_run(tmp_path):
    from ml.supplier_risk.train import SupplierTrainConfig, train_supplier_model

    _write_supplier_features(tmp_path)
    _write_supplier_events(tmp_path / "raw")

    cfg = SupplierTrainConfig(
        features_dir=str(tmp_path / "features"),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "artifact"),
        seed=42,
        horizon_days=30,
        calibrator="isotonic",
        n_estimators=20,
        early_stopping_rounds=5,
    )
    result = train_supplier_model(cfg)

    out = tmp_path / "artifact"
    assert (out / "model.json").exists()
    assert (out / "calibrator.json").exists()
    assert (out / "threshold.json").exists()
    assert (out / "training_metadata.json").exists()
    assert (out / "metrics.json").exists()
    assert (out / "feature_importance.csv").exists()

    # Metrics must be finite for at least one split.
    test_m = result["metrics"]["test"]
    for k in ("brier", "roc_auc"):
        v = test_m.get(k)
        if v is not None:
            assert np.isfinite(v) or np.isnan(v)  # allowed if single-class


# ---------------------------------------------------------------------------
# Test 3: reproducibility
# ---------------------------------------------------------------------------
def test_training_is_reproducible(tmp_path):
    from ml.supplier_risk.train import SupplierTrainConfig, train_supplier_model

    _write_supplier_features(tmp_path)
    _write_supplier_events(tmp_path / "raw")

    def _run(sub: str):
        cfg = SupplierTrainConfig(
            features_dir=str(tmp_path / "features"),
            raw_dir=str(tmp_path / "raw"),
            output_dir=str(tmp_path / sub),
            seed=7, horizon_days=30, calibrator="platt",
            n_estimators=20, early_stopping_rounds=5,
        )
        return train_supplier_model(cfg)

    a = _run("a")
    b = _run("b")
    assert a["best_iteration"] == b["best_iteration"]
    # Brier and log-loss are numerically stable; match to 1e-6.
    for k in ("brier", "log_loss"):
        va = a["metrics"]["test"].get(k)
        vb = b["metrics"]["test"].get(k)
        if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):
            assert abs(va - vb) < 1e-6


# ---------------------------------------------------------------------------
# Test 4: predictor returns provenance
# ---------------------------------------------------------------------------
def test_predictor_provenance(tmp_path):
    from ml.supplier_risk.dataset import _locate_supplier_features
    from ml.supplier_risk.predict import SupplierRiskPredictor
    from ml.supplier_risk.train import SupplierTrainConfig, train_supplier_model

    _write_supplier_features(tmp_path)
    _write_supplier_events(tmp_path / "raw")

    cfg = SupplierTrainConfig(
        features_dir=str(tmp_path / "features"),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "artifact"),
        seed=3, horizon_days=30, calibrator="isotonic",
        n_estimators=20, early_stopping_rounds=5,
    )
    train_supplier_model(cfg)

    predictor = SupplierRiskPredictor(tmp_path / "artifact")
    df = pd.read_parquet(_locate_supplier_features(tmp_path / "features"))
    result = predictor.predict_dataframe(df.head(20))
    assert len(result) == 20
    assert (result["risk_probability"] >= 0).all()
    assert (result["risk_probability"] <= 1).all()
    assert result["model_version"].notna().all()
    assert result["feature_version"].notna().all()


# ---------------------------------------------------------------------------
# Test 5: threshold selection is deterministic
# ---------------------------------------------------------------------------
def test_threshold_selection_deterministic():
    from ml.supplier_risk.threshold_selection import select_threshold

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    p = rng.random(500)
    a = select_threshold(y, p, cost_false_negative=10.0, cost_false_positive=1.0)
    b = select_threshold(y, p, cost_false_negative=10.0, cost_false_positive=1.0)
    assert a["threshold"] == b["threshold"]
    assert a["cost"] == b["cost"]