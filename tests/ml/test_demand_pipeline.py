"""End-to-end tiny-run tests for the demand LSTM pipeline.

Each test builds a minimal synthetic feature table in a temp directory,
runs a full train → evaluate → predict cycle, and asserts the pipeline
behaves correctly.

Runtime target: under ~30 seconds total on a laptop CPU.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures: build a tiny synthetic feature table
# ---------------------------------------------------------------------------
def _make_tiny_feature_table(tmp_path: Path, *, n_series: int = 3, n_days: int = 120) -> Path:
    """Write a minimal demand feature table to disk and return its directory.

    We build the file with the same columns `ml.demand.dataset` expects, using
    the FEATURE_PREFIXES the resolver looks for.
    """
    from data_platform.features.feature_registry import FEATURE_VERSION

    rng = np.random.default_rng(0)
    rows = []
    skus = [f"SKU-{i:03d}" for i in range(n_series)]
    warehouses = ["WH-A"]
    categories = ["CAT-A"]

    start = date(2022, 1, 1)
    for sku in skus:
        for wh in warehouses:
            for d in range(n_days):
                dt = start + timedelta(days=d)
                q = float(max(0.0, 10.0 + 5.0 * np.sin(2 * np.pi * d / 30) + rng.normal(0, 1)))
                row = {
                    "product_sku": sku,
                    "warehouse_code": wh,
                    "date": pd.Timestamp(dt),
                    "quantity": q,
                    # Encoder features.
                    "feat_demand_lag_1": q,
                    "feat_demand_lag_7": q,
                    "feat_demand_roll_mean_7": q,
                    "feat_demand_roll_std_7": 0.5,
                    "feat_demand_expand_mean": q,
                    "feat_promo_count_7": 0,
                    "feat_holiday_count_7": 0,
                    "feat_censored_count_7": 0,
                    # Decoder calendar features.
                    "feat_dow": pd.Timestamp(dt).weekday(),
                    "feat_dow_sin": np.sin(2 * np.pi * pd.Timestamp(dt).weekday() / 7),
                    "feat_dow_cos": np.cos(2 * np.pi * pd.Timestamp(dt).weekday() / 7),
                    "feat_dom": dt.day,
                    "feat_month": dt.month,
                    "feat_month_sin": np.sin(2 * np.pi * (dt.month - 1) / 12),
                    "feat_month_cos": np.cos(2 * np.pi * (dt.month - 1) / 12),
                    "feat_quarter": (dt.month - 1) // 3 + 1,
                    "feat_year": dt.year,
                    "feat_week": dt.isocalendar().week,
                    "feat_day_of_year": dt.timetuple().tm_yday,
                    "feat_doy_sin": np.sin(2 * np.pi * (dt.timetuple().tm_yday - 1) / 365.25),
                    "feat_doy_cos": np.cos(2 * np.pi * (dt.timetuple().tm_yday - 1) / 365.25),
                    "feat_is_month_start": int(dt.day <= 3),
                    "feat_is_month_end": int(dt.day >= 28),
                    "feat_is_weekend": int(pd.Timestamp(dt).weekday() >= 5),
                    "feat_is_holiday": 0,
                    "feat_days_to_holiday": 5,
                    # Static features carried in the feature table.
                    "category_code": categories[0],
                    "unit_cost": 5.0,
                    "unit_price": 12.0,
                    "base_demand": 10.0,
                    "lead_time_days": 7,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = tmp_path / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"demand_features_{FEATURE_VERSION}.parquet"
    df.to_parquet(file_path, index=False)
    return out_dir


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def test_build_demand_datasets_smoke(tmp_path):
    from ml.demand.dataset import build_demand_datasets

    features_dir = _make_tiny_feature_table(tmp_path)
    bundle = build_demand_datasets(
        features_dir,
        lookback=14, horizon=5,
        val_fraction=0.15, test_fraction=0.15,
    )
    assert bundle.n_products >= 2          # 3 SKUs + unknown slot
    assert bundle.n_warehouses >= 2
    assert len(bundle.train) > 0
    assert len(bundle.encoder_columns) > 0
    assert len(bundle.decoder_columns) > 0
    # Scaler means should be finite.
    assert np.isfinite(bundle.scalers.enc_mean).all()
    assert np.isfinite(bundle.scalers.enc_std).all()


def test_windowed_dataset_shapes(tmp_path):
    from ml.demand.dataset import build_demand_datasets

    features_dir = _make_tiny_feature_table(tmp_path)
    bundle = build_demand_datasets(
        features_dir, lookback=10, horizon=4,
    )
    sample = bundle.train[0]
    assert sample["encoder"].shape == (10, len(bundle.encoder_columns))
    assert sample["decoder"].shape == (4, len(bundle.decoder_columns))
    assert sample["target"].shape == (4,)
    assert sample["static_cat"].shape == (3,)
    assert sample["static_num"].shape == (len(bundle.static_num_columns),)


# ---------------------------------------------------------------------------
# Training: tiny 2-epoch run, checks artifacts exist
# ---------------------------------------------------------------------------
def test_train_tiny_run_produces_artifacts(tmp_path):
    from ml.demand.train import TrainConfig, train_demand_model

    features_dir = _make_tiny_feature_table(tmp_path, n_series=2, n_days=90)
    out_dir = tmp_path / "artifacts"
    cfg = TrainConfig(
        features_dir=str(features_dir),
        output_dir=str(out_dir),
        seed=42,
        lookback=10,
        horizon=4,
        batch_size=16,
        epochs=2,
        lr=1e-3,
        early_stop_patience=2,
        num_workers=0,
        device="cpu",
        max_series=10,
        quantiles=None,   # point model — fast
    )
    meta = train_demand_model(cfg)

    # Files
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "scalers.npz").exists()
    assert (out_dir / "training_metadata.json").exists()
    assert (out_dir / "metrics.json").exists()

    # Metadata
    assert meta["feature_version"].startswith("v1_")
    assert "val_metrics" in meta
    assert "test_metrics" in meta
    # Metrics must be finite.
    for k in ("mae", "rmse", "wape"):
        v = meta["val_metrics"].get(k)
        if v is not None:
            assert np.isfinite(v)


def test_training_is_reproducible(tmp_path):
    """Same seed, same data → identical test metrics."""
    from ml.demand.train import TrainConfig, train_demand_model

    features_dir = _make_tiny_feature_table(tmp_path, n_series=2, n_days=80)

    def _run(out_sub: str) -> dict:
        cfg = TrainConfig(
            features_dir=str(features_dir),
            output_dir=str(tmp_path / out_sub),
            seed=7, lookback=10, horizon=4,
            batch_size=16, epochs=1, lr=1e-3,
            early_stop_patience=2, num_workers=0,
            device="cpu", max_series=10, quantiles=None,
        )
        return train_demand_model(cfg)

    m1 = _run("run_a")
    m2 = _run("run_b")

    # Best validation loss identical to 1e-5 (allows for float64→float32 noise).
    assert abs(m1["best_val_loss"] - m2["best_val_loss"]) < 1e-5
    assert abs(m1["val_metrics"]["mae"] - m2["val_metrics"]["mae"]) < 1e-4


# ---------------------------------------------------------------------------
# Predictor: single-series shape + provenance
# ---------------------------------------------------------------------------
def test_predictor_returns_provenance_and_finite_values(tmp_path):
    from ml.demand.predict import DemandPredictor
    from ml.demand.train import TrainConfig, train_demand_model

    features_dir = _make_tiny_feature_table(tmp_path, n_series=2, n_days=90)
    ckpt_dir = tmp_path / "artifact"
    cfg = TrainConfig(
        features_dir=str(features_dir),
        output_dir=str(ckpt_dir),
        seed=1, lookback=10, horizon=4,
        batch_size=16, epochs=2, lr=1e-3,
        early_stop_patience=2, num_workers=0,
        device="cpu", max_series=10, quantiles=None,
    )
    train_demand_model(cfg)

    predictor = DemandPredictor(ckpt_dir, device="cpu")

    # Load the feature frame.
    feature_file = next(Path(features_dir).glob("demand_features_*.parquet"))
    df = pd.read_parquet(feature_file)

    sku = df["product_sku"].iloc[0]
    wh = df["warehouse_code"].iloc[0]

    fc = predictor.predict(df, product_sku=sku, warehouse_code=wh)
    assert len(fc.point) == predictor.horizon
    assert len(fc.forecast_dates) == predictor.horizon
    assert np.isfinite(fc.point).all()
    assert fc.feature_version.startswith("v1_")
    assert fc.model_version  # non-empty
    assert fc.seed == 1


# ---------------------------------------------------------------------------
# Determinism of evaluation
# ---------------------------------------------------------------------------
def test_evaluation_is_reproducible(tmp_path):
    from ml.demand.evaluate import evaluate_checkpoint
    from ml.demand.train import TrainConfig, train_demand_model

    features_dir = _make_tiny_feature_table(tmp_path, n_series=2, n_days=90)
    ckpt_dir = tmp_path / "artifact"
    cfg = TrainConfig(
        features_dir=str(features_dir), output_dir=str(ckpt_dir),
        seed=2, lookback=10, horizon=4,
        batch_size=16, epochs=1, lr=1e-3,
        early_stop_patience=2, num_workers=0,
        device="cpu", max_series=10, quantiles=None,
    )
    train_demand_model(cfg)

    out_a = tmp_path / "eval_a"
    out_b = tmp_path / "eval_b"
    r1 = evaluate_checkpoint(ckpt_dir, features_dir, output_dir=out_a,
                             run_name="a", batch_size=16, device="cpu")
    r2 = evaluate_checkpoint(ckpt_dir, features_dir, output_dir=out_b,
                             run_name="b", batch_size=16, device="cpu")

    # Test-set WAPE must match exactly — the model and data are identical.
    assert abs(r1["metrics"]["test"]["wape"] - r2["metrics"]["test"]["wape"]) < 1e-9
    assert abs(r1["metrics"]["test"]["mae"] - r2["metrics"]["test"]["mae"]) < 1e-9