"""Training entry point for supplier risk.

Deterministic given the seed. Saves:
    artifacts/supplier/<mode>/model.json            (XGBoost booster)
    artifacts/supplier/<mode>/calibrator.json       (calibration curve params)
    artifacts/supplier/<mode>/training_metadata.json
    artifacts/supplier/<mode>/metrics.json
    artifacts/supplier/<mode>/feature_importance.csv
    artifacts/supplier/<mode>/threshold.json

Calibration is fit on validation only and applied to test. Threshold selection
also uses only validation predictions (never test).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import get_settings
from core.deterministic import seed_everything
from core.logging import configure_logging, get_logger
from ml.supplier_risk.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    fit_calibrator,
    save_calibrator,
)
from ml.supplier_risk.dataset import build_supplier_dataset
from ml.supplier_risk.evaluate import evaluate_binary
from ml.supplier_risk.model import SupplierModelConfig, SupplierRiskModel
from ml.supplier_risk.threshold_selection import (
    select_threshold,
    save_threshold,
)

log = get_logger(__name__)


@dataclass
class SupplierTrainConfig:
    features_dir: str
    output_dir: str
    raw_dir: Optional[str] = None
    seed: int = 42
    horizon_days: int = 30
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    max_rows: Optional[int] = None
    calibrator: str = "isotonic"           # "isotonic" | "platt" | "none"
    # Cost matrix for threshold selection (validation only).
    cost_false_negative: float = 10.0
    cost_false_positive: float = 1.0
    # Model hyperparameters.
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30

    def to_dict(self) -> dict:
        return asdict(self)


def train_supplier_model(cfg: SupplierTrainConfig) -> dict:
    configure_logging()
    settings = get_settings()
    seed_everything(cfg.seed, deterministic=settings.DETERMINISTIC)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("supplier.train.start", extra=cfg.to_dict())

    bundle = build_supplier_dataset(
        cfg.features_dir,
        raw_dir=cfg.raw_dir,
        horizon_days=cfg.horizon_days,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        max_rows=cfg.max_rows,
    )

    if len(bundle.X_train) == 0:
        raise ValueError("Empty training set.")

    # Convert to numpy for XGBoost.
    X_train = bundle.X_train.to_numpy(dtype=np.float32)
    y_train = bundle.y_train.to_numpy(dtype=np.int32)
    X_val = bundle.X_val.to_numpy(dtype=np.float32)
    y_val = bundle.y_val.to_numpy(dtype=np.int32)
    X_test = bundle.X_test.to_numpy(dtype=np.float32)
    y_test = bundle.y_test.to_numpy(dtype=np.int32)

    model_cfg = SupplierModelConfig(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_weight=cfg.min_child_weight,
        reg_lambda=cfg.reg_lambda,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    model = SupplierRiskModel(model_cfg)

    log.info("supplier.train.fitting", extra={
        "n_train": len(X_train), "n_features": X_train.shape[1],
        "positive_rate": float(np.mean(y_train)),
    })
    model.fit(X_train, y_train, X_val, y_val)
    log.info("supplier.train.fitted", extra={"best_iteration": model.best_iteration})

    # Raw (uncalibrated) probabilities.
    p_val_raw = model.predict_proba(X_val)
    p_test_raw = model.predict_proba(X_test)
    p_train_raw = model.predict_proba(X_train)

    # Calibration (fit on val only, applied to test).
    calibrator = None
    if cfg.calibrator == "isotonic":
        calibrator = fit_calibrator("isotonic", p_val_raw, y_val)
    elif cfg.calibrator == "platt":
        calibrator = fit_calibrator("platt", p_val_raw, y_val)
    # else None

    if calibrator is not None:
        p_val = calibrator.transform(p_val_raw)
        p_test = calibrator.transform(p_test_raw)
        p_train = calibrator.transform(p_train_raw)
    else:
        p_val, p_test, p_train = p_val_raw, p_test_raw, p_train_raw

    # Threshold selection on validation only.
    thr_result = select_threshold(
        y_val, p_val,
        cost_false_negative=cfg.cost_false_negative,
        cost_false_positive=cfg.cost_false_positive,
    )
    threshold = thr_result["threshold"]

    # Metrics.
    metrics_train = evaluate_binary(y_train, p_train, threshold=threshold)
    metrics_val = evaluate_binary(y_val, p_val, threshold=threshold)
    metrics_test = evaluate_binary(y_test, p_test, threshold=threshold)

    # Persist model, calibrator, threshold.
    model.save(str(out_dir / "model.json"))
    if calibrator is not None:
        save_calibrator(calibrator, out_dir / "calibrator.json")
    save_threshold(thr_result, out_dir / "threshold.json")

    # Feature importance.
    importance = model.feature_importance(bundle.feature_columns)
    with (out_dir / "feature_importance.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "gain"])
        for name, gain in sorted(importance.items(), key=lambda kv: -kv[1]):
            w.writerow([name, gain])

    metadata = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "feature_version": bundle.feature_version,
        "seed": cfg.seed,
        "horizon_days": cfg.horizon_days,
        "calibrator": cfg.calibrator,
        "best_iteration": model.best_iteration,
        "model_config": model_cfg.to_dict(),
        "train_config": cfg.to_dict(),
        "split_dates": bundle.split_dates,
        "dataset_summary": bundle.summary(),
        "feature_columns": bundle.feature_columns,
        "threshold": thr_result,
    }
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps({
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        }, indent=2, default=str), encoding="utf-8"
    )

    log.info("supplier.train.done", extra={
        "val_auc": metrics_val.get("roc_auc"),
        "test_auc": metrics_test.get("roc_auc"),
        "test_pr_auc": metrics_test.get("pr_auc"),
        "threshold": threshold,
    })

    return {
        "trained_at": metadata["trained_at"],
        "feature_version": bundle.feature_version,
        "seed": cfg.seed,
        "best_iteration": model.best_iteration,
        "threshold": threshold,
        "dataset_summary": bundle.summary(),
        "metrics": {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        },
        "output_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> SupplierTrainConfig:
    p = argparse.ArgumentParser(description="Train the supplier risk classifier.")
    p.add_argument("--features", required=True,
                   help="Directory with supplier_features_*.parquet")
    p.add_argument("--raw", default=None,
                   help="Directory containing supplier_events parquet")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon-days", type=int, default=30)
    p.add_argument("--calibrator", choices=["isotonic", "platt", "none"], default="isotonic")
    p.add_argument("--cost-fn", type=float, default=10.0)
    p.add_argument("--cost-fp", type=float, default=1.0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--early-stopping-rounds", type=int, default=30)
    args = p.parse_args()

    return SupplierTrainConfig(
        features_dir=args.features,
        raw_dir=args.raw,
        output_dir=args.output,
        seed=args.seed,
        horizon_days=args.horizon_days,
        calibrator=args.calibrator,
        cost_false_negative=args.cost_fn,
        cost_false_positive=args.cost_fp,
        max_rows=args.max_rows,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    result = train_supplier_model(cfg)
    print(json.dumps(result["metrics"], indent=2, default=str))