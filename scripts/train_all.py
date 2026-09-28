"""CLI: orchestrate all model training.

Currently trains the demand LSTM, supplier-risk (XGBoost), and inventory (DQN) targets.
The CLI is structured so each sub-trainer is independent and records its own
provenance and registers the resulting model in the registry.

Usage:
    python scripts/train_all.py --mode dev --model demand
    python scripts/train_all.py --mode research --model demand --device cuda
    python scripts/train_all.py --mode dev --model all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402
from ml.demand.train import TrainConfig, train_demand_model  # noqa: E402

log = get_logger(__name__)


def _register_model(spec_dict: dict) -> Optional[int]:
    """Register a trained model into the registry. Returns the DB row id or None on failure."""
    try:
        from core.database import session_scope
        from model_registry.registry import ModelVersionSpec, Registry

        spec = ModelVersionSpec(**spec_dict)
        with session_scope() as session:
            reg = Registry(session)
            row = reg.register(spec)
            return int(row.id)
    except Exception as e:
        log.warning("train_all.registry_skipped", extra={"error": str(e)})
        return None


def _demand_cfg_for_mode(
    mode: str,
    features_dir: str,
    output_dir: str,
    seed: int,
    device: str,
    max_series: int | None,
) -> TrainConfig:
    """Choose sensible hyperparameters per dataset mode.

    Small datasets use smaller windows to have enough samples; large datasets
    use the full research configuration.
    """
    if mode == "dev":
        return TrainConfig(
            features_dir=features_dir,
            output_dir=output_dir,
            seed=seed,
            lookback=30,
            horizon=7,
            batch_size=128,
            epochs=5,
            lr=1e-3,
            early_stop_patience=2,
            device=device,
            max_series=max_series,
            quantiles=(0.1, 0.5, 0.9),
        )
    if mode == "research":
        return TrainConfig(
            features_dir=features_dir,
            output_dir=output_dir,
            seed=seed,
            lookback=90,
            horizon=30,
            batch_size=256,
            epochs=30,
            lr=1e-3,
            early_stop_patience=5,
            device=device,
            max_series=max_series,
            quantiles=(0.1, 0.5, 0.9),
        )
    if mode == "stress":
        return TrainConfig(
            features_dir=features_dir,
            output_dir=output_dir,
            seed=seed,
            lookback=120,
            horizon=30,
            batch_size=512,
            epochs=20,
            lr=8e-4,
            early_stop_patience=4,
            device=device,
            max_series=max_series,
            quantiles=(0.1, 0.5, 0.9),
        )
    raise ValueError(f"Unknown mode: {mode}")


def train_demand(
    mode: str, seed: int, device: str, max_series: int | None
) -> Dict[str, Any]:
    features_dir = f"datasets/curated/{mode}/features"
    output_dir = f"artifacts/demand/{mode}"
    if not Path(features_dir).exists():
        raise FileNotFoundError(
            f"Missing feature directory: {features_dir}. "
            f"Run `python scripts/build_features.py --mode {mode}` first."
        )
    cfg = _demand_cfg_for_mode(mode, features_dir, output_dir, seed, device, max_series)
    log.info("train_all.demand.start", extra=cfg.to_dict())
    meta = train_demand_model(cfg)

    # Register the freshly trained model.
    from data_platform.features.feature_registry import FEATURE_VERSION
    from model_registry.versioning import compute_dataset_version

    dataset_version = compute_dataset_version(
        mode=mode,
        seed=seed,
        row_counts={"n_series": meta.get("n_params", 0)},  # structural placeholder
    )
    register_payload = {
        "model_name": "demand_lstm_global",
        "model_family": "demand_lstm",
        "framework": "pytorch",
        "hyperparameters": meta["model_config"],
        "metrics": {"val": meta["val_metrics"], "test": meta["test_metrics"]},
        "dataset_version": dataset_version,
        "feature_version": FEATURE_VERSION,
        "artifact_dir": output_dir,
        "trained_by": "scripts.train_all",
        "training_data_summary": {
            "n_products": meta.get("n_products"),
            "n_warehouses": meta.get("n_warehouses"),
            "split_dates": meta.get("split_dates"),
        },
        "notes": [f"trained_on_mode={mode}"],
    }
    reg_id = _register_model(register_payload)
    meta["registry_id"] = reg_id

    return {
        "model": "demand",
        "output_dir": output_dir,
        "best_val_loss": meta.get("best_val_loss"),
        "val_metrics": meta.get("val_metrics"),
        "test_metrics": meta.get("test_metrics"),
        "feature_version": meta.get("feature_version"),
        "registry_id": reg_id,
    }


# --- Supplier ---
def train_supplier(mode: str, seed: int, max_rows: int | None) -> Dict[str, Any]:
    from ml.supplier_risk.train import SupplierTrainConfig, train_supplier_model

    features_dir = f"datasets/curated/{mode}/features"
    raw_dir = f"datasets/raw/{mode}"
    output_dir = f"artifacts/supplier/{mode}"
    if not Path(features_dir).exists():
        raise FileNotFoundError(
            f"Missing feature directory: {features_dir}. "
            f"Run `python scripts/build_features.py --mode {mode}` first."
        )
    cfg = SupplierTrainConfig(
        features_dir=features_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        seed=seed,
        horizon_days=30,
        calibrator="isotonic",
        max_rows=max_rows,
    )
    log.info("train_all.supplier.start", extra=cfg.to_dict())
    result = train_supplier_model(cfg)

    from data_platform.features.feature_registry import FEATURE_VERSION
    from model_registry.versioning import compute_dataset_version

    register_payload = {
        "model_name": "supplier_risk_xgb",
        "model_family": "supplier_xgb",
        "framework": "xgboost",
        "hyperparameters": result.get("model_config") or {},
        "metrics": result["metrics"],
        "dataset_version": compute_dataset_version(
            mode=mode,
            seed=seed,
            row_counts={
                "train": result["dataset_summary"]["n_train"],
                "val": result["dataset_summary"]["n_val"],
                "test": result["dataset_summary"]["n_test"],
            },
        ),
        "feature_version": FEATURE_VERSION,
        "artifact_dir": output_dir,
        "trained_by": "scripts.train_all",
        "training_data_summary": result["dataset_summary"],
    }
    result["registry_id"] = _register_model(register_payload)

    return {
        "model": "supplier",
        "output_dir": output_dir,
        "threshold": result["threshold"],
        "metrics": result["metrics"],
        "feature_version": result["feature_version"],
        "registry_id": result["registry_id"],
    }


# --- Inventory ---
def train_inventory(
    mode: str, seed: int, device: str, max_series: int | None
) -> Dict[str, Any]:
    from ml.inventory.train import DQNTrainConfig, train_inventory_dqn

    features_dir = f"datasets/curated/{mode}/features"
    output_dir = f"artifacts/inventory/{mode}"
    supplier_dir = f"artifacts/supplier/{mode}"
    if not Path(features_dir).exists():
        raise FileNotFoundError(f"Missing feature directory: {features_dir}.")

    # Adjust episode count by mode.
    n_episodes = {"dev": 30, "research": 200, "stress": 150}[mode]
    episode_days = {"dev": 30, "research": 60, "stress": 60}[mode]
    max_series_cfg = (
        max_series if max_series is not None else (5 if mode == "dev" else 10)
    )

    cfg = DQNTrainConfig(
        features_dir=features_dir,
        output_dir=output_dir,
        supplier_artifacts_dir=supplier_dir if Path(supplier_dir).exists() else None,
        seed=seed,
        episode_days=episode_days,
        n_episodes=n_episodes,
        max_series=max_series_cfg,
        device=device,
    )
    log.info(
        "train_all.inventory.start",
        extra={
            "mode": mode,
            "n_episodes": n_episodes,
            "episode_days": episode_days,
        },
    )
    meta = train_inventory_dqn(cfg)

    from data_platform.features.feature_registry import FEATURE_VERSION
    from model_registry.versioning import compute_dataset_version

    register_payload = {
        "model_name": "inventory_dqn",
        "model_family": "inventory_dqn",
        "framework": "pytorch",
        "hyperparameters": meta["q_network_config"],
        "metrics": {
            "final_eval_cost": meta["final_eval_cost"],
            "best_eval_cost": meta["best_eval_cost"],
        },
        "dataset_version": compute_dataset_version(
            mode=mode, seed=seed, row_counts={"n_series": meta["n_series"]}
        ),
        "feature_version": FEATURE_VERSION,
        "artifact_dir": output_dir,
        "trained_by": "scripts.train_all",
        "training_data_summary": {
            "n_series": meta["n_series"],
            "episode_days": meta["train_config"]["episode_days"],
        },
    }
    meta["registry_id"] = _register_model(register_payload)

    return {
        "model": "inventory",
        "output_dir": output_dir,
        "best_eval_cost": meta["best_eval_cost"],
        "final_eval_cost": meta["final_eval_cost"],
        "registry_id": meta["registry_id"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Train all models.")
    p.add_argument("--mode", choices=["dev", "research", "stress"], default="dev")
    p.add_argument(
        "--model",
        choices=["demand", "supplier", "inventory", "all"],
        default="demand",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed. Defaults to settings.RANDOM_SEED.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Cap the number of series (useful for smoke tests).",
    )
    args = p.parse_args()

    configure_logging()
    settings = get_settings()
    seed = args.seed if args.seed is not None else settings.RANDOM_SEED

    summary: Dict[str, Any] = {"mode": args.mode, "seed": seed, "models": {}}

    if args.model in ("demand", "all"):
        summary["models"]["demand"] = train_demand(
            args.mode, seed, args.device, args.max_series
        )

    if args.model in ("supplier", "all"):
        summary["models"]["supplier"] = train_supplier(
            args.mode, seed, args.max_series
        )

    if args.model in ("inventory", "all"):
        summary["models"]["inventory"] = train_inventory(
            args.mode, seed, args.device, args.max_series
        )

    out_path = Path("reports/metrics") / f"train_all_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

