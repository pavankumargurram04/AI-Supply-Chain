"""CLI: evaluate every trained model.

Currently evaluates the demand LSTM. Later phases extend this with supplier
and inventory evaluation.

Usage:
    python scripts/evaluate_all.py --mode dev --model demand
    python scripts/evaluate_all.py --mode research --model all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logging import configure_logging, get_logger  # noqa: E402
from ml.demand.evaluate import evaluate_checkpoint  # noqa: E402

log = get_logger(__name__)


def evaluate_demand(mode: str, device: str, max_series: int | None) -> Dict[str, Any]:
    ckpt_dir = f"artifacts/demand/{mode}"
    features_dir = f"datasets/curated/{mode}/features"
    if not Path(ckpt_dir).exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_dir}. Run `python scripts/train_all.py --mode {mode}` first."
        )
    payload = evaluate_checkpoint(
        ckpt_dir, features_dir,
        output_dir="reports/metrics",
        run_name=mode,
        device=device,
        max_series=max_series,
    )
    return {
        "model": "demand",
        "run_name": mode,
        "metrics_json": f"reports/metrics/demand_{mode}_metrics.json",
        "metrics": payload["metrics"],
    }
def evaluate_supplier(mode: str) -> Dict[str, Any]:
    from ml.supplier_risk.predict import SupplierRiskPredictor
    from ml.supplier_risk.dataset import _locate_supplier_features

    ckpt_dir = Path(f"artifacts/supplier/{mode}")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No supplier checkpoint at {ckpt_dir}.")

    # Re-run evaluation from the training metadata (which already contains
    # train / val / test metrics for the exact split the model was trained on).
    import json
    meta = json.loads((ckpt_dir / "training_metadata.json").read_text())
    metrics = json.loads((ckpt_dir / "metrics.json").read_text())

    # Feature importance snapshot (top 20).
    import csv
    top_feats = []
    with (ckpt_dir / "feature_importance.csv").open() as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= 20:
                break
            top_feats.append({"feature": row["feature"], "gain": float(row["gain"])})

    return {
        "model": "supplier",
        "run_name": mode,
        "threshold": meta.get("threshold", {}).get("threshold"),
        "metrics": metrics,
        "top_features": top_feats,
        "model_config": meta["model_config"],
    }
def evaluate_inventory(mode: str, max_series: int | None) -> Dict[str, Any]:
    from ml.inventory.evaluate import run_inventory_eval

    ckpt_dir = Path(f"artifacts/inventory/{mode}")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No inventory checkpoint at {ckpt_dir}.")
    features_dir = f"datasets/curated/{mode}/features"

    result = run_inventory_eval(
        features_dir=features_dir,
        checkpoint_dir=ckpt_dir,
        output_dir="reports/metrics",
        run_name=mode,
        max_series=max_series if max_series is not None else (3 if mode == "dev" else 5),
        episode_days=60 if mode != "dev" else 30,
    )
    return {
        "model": "inventory",
        "run_name": mode,
        "aggregate": result["aggregate"],
        "json_path": f"reports/metrics/inventory_eval_{mode}.json",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate all trained models.")
    p.add_argument("--mode", choices=["dev", "research", "stress"], default="dev")
    p.add_argument("--model", choices=["demand", "supplier", "inventory", "all"], default="demand")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-series", type=int, default=None)
    args = p.parse_args()

    configure_logging()
    summary: Dict[str, Any] = {"mode": args.mode, "models": {}}

    if args.model in ("demand", "all"):
        summary["models"]["demand"] = evaluate_demand(args.mode, args.device, args.max_series)

    if args.model in ("supplier", "all"):
        summary["models"]["supplier"] = evaluate_supplier(args.mode)

    if args.model in ("inventory", "all"):
        summary["models"]["inventory"] = evaluate_inventory(args.mode, args.max_series)    

    out_path = Path("reports/metrics") / f"evaluate_all_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())