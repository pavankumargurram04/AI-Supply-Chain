"""CLI: rolling-origin backtest orchestration.

Usage:
    python scripts/backtest_all.py --mode dev --model demand --folds 3 --max-series 10
    python scripts/backtest_all.py --mode research --model all --folds 6
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
from ml.demand.backtesting import run_backtest  # noqa: E402

log = get_logger(__name__)


def backtest_demand(mode: str, folds: int, max_series: int | None, device: str) -> Dict[str, Any]:
    ckpt_dir = f"artifacts/demand/{mode}"
    features_dir = f"datasets/curated/{mode}/features"
    if not Path(ckpt_dir).exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_dir}.")

    result = run_backtest(
        ckpt_dir, features_dir,
        output_dir="reports/metrics",
        run_name=mode,
        folds=folds,
        max_series=max_series,
        device=device,
    )
    return {
        "model": "demand",
        "run_name": mode,
        "n_folds": result["n_folds"],
        "aggregate": result["aggregate"],
        "json_path": f"reports/metrics/demand_{mode}_backtest.json",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Rolling-origin backtest.")
    p.add_argument("--mode", choices=["dev", "research", "stress"], default="dev")
    p.add_argument("--model", choices=["demand", "supplier", "inventory", "all"], default="demand")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-series", type=int, default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    configure_logging()
    summary: Dict[str, Any] = {"mode": args.mode, "models": {}}

    if args.model in ("demand", "all"):
        summary["models"]["demand"] = backtest_demand(
            args.mode, args.folds, args.max_series, args.device
        )

    if args.model in ("supplier", "inventory", "all"):
        log.warning("backtest_all.not_implemented", extra={"model": args.model})

    out_path = Path("reports/metrics") / f"backtest_all_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())