"""End-to-end ML test.

Runs the demand training pipeline on a tiny synthetic table, then verifies
that the predictor produces a shape-consistent forecast and that residuals
can be extracted. Requires only torch CPU and numpy.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_tiny_features(root: Path, n_series: int = 2, n_days: int = 90) -> Path:
    from data_platform.features.feature_registry import FEATURE_VERSION

    rng = np.random.default_rng(0)
    start = date(2022, 1, 1)
    rows = []
    for s_i in range(n_series):
        sku = f"SKU-{s_i:03d}"
        for d in range(n_days):
            dt = start + timedelta(days=d)
            q = float(max(0.0, 20.0 + rng.normal(0, 2)))
            rows.append({
                "product_sku": sku, "warehouse_code": "WH-A",
                "date": pd.Timestamp(dt), "quantity": q,
                "feat_demand_lag_1": q, "feat_demand_lag_7": q,
                "feat_demand_roll_mean_7": q, "feat_demand_roll_std_7": 1.0,
                "feat_demand_expand_mean": q,
                "feat_promo_count_7": 0, "feat_holiday_count_7": 0,
                "feat_censored_count_7": 0,
                "feat_dow": pd.Timestamp(dt).weekday(),
                "feat_dow_sin": 0.0, "feat_dow_cos": 1.0,
                "feat_dom": dt.day, "feat_month": dt.month,
                "feat_month_sin": 0.0, "feat_month_cos": 1.0,
                "feat_quarter": 1, "feat_year": dt.year,
                "feat_week": dt.isocalendar().week,
                "feat_day_of_year": dt.timetuple().tm_yday,
                "feat_doy_sin": 0.0, "feat_doy_cos": 1.0,
                "feat_is_month_start": 0, "feat_is_month_end": 0,
                "feat_is_weekend": int(pd.Timestamp(dt).weekday() >= 5),
                "feat_is_holiday": 0, "feat_days_to_holiday": 5,
                "category_code": "CAT-A", "unit_cost": 10.0, "unit_price": 25.0,
                "base_demand": 20.0, "lead_time_days": 7,
            })
    fdir = root / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        fdir / f"demand_features_{FEATURE_VERSION}.parquet", index=False,
    )
    return fdir


def test_training_predictor_roundtrip(tmp_path):
    """Train a tiny model, load it, forecast one series, check shape + provenance."""
    from ml.demand.train import TrainConfig, train_demand_model
    from ml.demand.predict import DemandPredictor

    fdir = _write_tiny_features(tmp_path)
    ckpt_dir = tmp_path / "artifact"

    cfg = TrainConfig(
        features_dir=str(fdir),
        output_dir=str(ckpt_dir),
        seed=1, lookback=10, horizon=5,
        batch_size=16, epochs=1, lr=1e-3,
        early_stop_patience=1, num_workers=0,
        device="cpu", max_series=10, quantiles=None,
    )
    meta = train_demand_model(cfg)
    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "scalers.npz").exists()
    assert meta["feature_version"].startswith("v1_")

    predictor = DemandPredictor(ckpt_dir, device="cpu")
    feature_file = next(Path(fdir).glob("demand_features_*.parquet"))
    df = pd.read_parquet(feature_file)

    sku = df["product_sku"].iloc[0]
    wh = df["warehouse_code"].iloc[0]
    fc = predictor.predict(df, product_sku=sku, warehouse_code=wh)

    assert len(fc.point) == 5
    assert len(fc.forecast_dates) == 5
    assert np.isfinite(fc.point).all()
    assert fc.feature_version.startswith("v1_")
    assert fc.model_version


def test_residual_extraction(tmp_path):
    from ml.demand.train import TrainConfig, train_demand_model
    from ml.demand.predict import DemandPredictor
    from ml.demand.residuals import extract_residuals

    fdir = _write_tiny_features(tmp_path, n_series=1, n_days=60)
    ckpt_dir = tmp_path / "artifact"

    cfg = TrainConfig(
        features_dir=str(fdir), output_dir=str(ckpt_dir),
        seed=2, lookback=8, horizon=3,
        batch_size=16, epochs=1, lr=1e-3,
        early_stop_patience=1, num_workers=0,
        device="cpu", max_series=10, quantiles=None,
    )
    train_demand_model(cfg)
    predictor = DemandPredictor(ckpt_dir, device="cpu")

    feature_file = next(Path(fdir).glob("demand_features_*.parquet"))
    df = pd.read_parquet(feature_file)
    sku = df["product_sku"].iloc[0]
    wh = df["warehouse_code"].iloc[0]

    residuals = extract_residuals(predictor, df, max_series=1, stride=5)
    if not residuals.empty:
        assert {"product_sku", "warehouse_code", "horizon", "residual"} <= set(residuals.columns)