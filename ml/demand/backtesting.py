"""Rolling-origin backtest for the demand model.

For each cutoff date t in the backtest window, we refit nothing (the model is
frozen) and produce a horizon forecast using data up to t only. Metrics are
aggregated per cutoff, giving a time-resolved view of model accuracy.

This module reuses the same scaler + model as training; it does NOT refit.

Usage:
    python -m ml.demand.backtesting \
        --checkpoint artifacts/demand/dev_smoke \
        --features datasets/curated/dev/features \
        --output reports/metrics \
        --run-name dev_bt \
        --folds 5
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from core.logging import get_logger
from data_platform.features.feature_registry import FEATURE_VERSION
from data_platform.splits.time_series_split import compute_split_dates
from ml.demand.evaluate import compute_metrics
from ml.demand.predict import DemandPredictor

log = get_logger(__name__)


@dataclass
class BacktestFold:
    fold: int
    origin_date: str
    n_series: int
    metrics: dict


def _per_series_map(df: pd.DataFrame) -> List[tuple[str, str]]:
    keys = df[["product_sku", "warehouse_code"]].drop_duplicates().values.tolist()
    return [tuple(k) for k in keys]


def run_backtest(
    checkpoint_dir: str | Path,
    features_dir: str | Path,
    *,
    output_dir: str | Path,
    run_name: str,
    folds: int = 5,
    max_series: Optional[int] = None,
    device: str = "auto",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load feature table.
    features_path = next(Path(features_dir).glob("demand_features_*.parquet"), None)
    if features_path is None:
        raise FileNotFoundError(f"No demand feature file in {features_dir}")
    df = pd.read_parquet(features_path)

    # Determine the test window from chronological split.
    split = compute_split_dates(df, date_col="date")
    test_start = pd.Timestamp(split.test_start)
    df_test_window = df[pd.to_datetime(df["date"]) >= test_start].copy()
    if df_test_window.empty:
        raise ValueError("Test window is empty — cannot run backtest.")

    test_dates = sorted(pd.to_datetime(df_test_window["date"]).dt.normalize().unique())
    if len(test_dates) <= 1:
        raise ValueError("Test window too short for a backtest.")

    # Pick `folds` cutoff dates evenly across the test window.
    n_dates = len(test_dates)
    if folds >= n_dates:
        cutoffs = [pd.Timestamp(d) for d in test_dates[:-1]]
    else:
        step = max(1, n_dates // folds)
        cutoffs = [pd.Timestamp(test_dates[i]) for i in range(0, n_dates - 1, step)][:folds]

    predictor = DemandPredictor(checkpoint_dir, device=device)
    series_list = _per_series_map(df)
    if max_series is not None:
        series_list = series_list[:max_series]

    results: List[BacktestFold] = []
    per_series_rows: List[dict] = []

    for fi, cutoff in enumerate(cutoffs):
        preds_all: List[np.ndarray] = []
        targets_all: List[np.ndarray] = []

        for sku, wh in series_list:
            try:
                fc = predictor.predict(
                    df, product_sku=sku, warehouse_code=wh, origin_date=cutoff,
                )
            except Exception as e:
                log.warning("backtest.fold.series_failed", extra={
                    "sku": sku, "wh": wh, "cutoff": str(cutoff), "err": str(e),
                })
                continue

            y_true = df[
                (df["product_sku"] == sku)
                & (df["warehouse_code"] == wh)
                & (pd.to_datetime(df["date"]) > cutoff)
            ].sort_values("date").head(predictor.horizon)["quantity"].to_numpy(dtype=float)

            if len(y_true) < predictor.horizon:
                continue

            preds_all.append(np.array(fc.point))
            targets_all.append(y_true)

        if not preds_all:
            log.warning("backtest.fold.empty", extra={"fold": fi, "cutoff": str(cutoff)})
            continue

        P = np.stack(preds_all, axis=0)
        T = np.stack(targets_all, axis=0)
        metrics = compute_metrics(P, T)
        metrics["origin_date"] = cutoff.date().isoformat()
        metrics["fold"] = fi
        results.append(BacktestFold(
            fold=fi,
            origin_date=cutoff.date().isoformat(),
            n_series=int(P.shape[0]),
            metrics=metrics,
        ))

        # Per-series detail rows for this fold.
        for i, (sku, wh) in enumerate(series_list[: P.shape[0]]):
            per_series_rows.append({
                "fold": fi,
                "origin_date": cutoff.date().isoformat(),
                "product_sku": sku,
                "warehouse_code": wh,
                "mae": float(np.mean(np.abs(P[i] - T[i]))),
                "rmse": float(np.sqrt(np.mean((P[i] - T[i]) ** 2))),
                "bias": float(np.mean(P[i] - T[i])),
            })

    # Aggregate.
    if not results:
        raise RuntimeError("No backtest folds produced results.")

    agg = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_dir),
        "feature_version": FEATURE_VERSION,
        "n_folds": len(results),
        "folds": [r.metrics for r in results],
        "aggregate": {
            "mae_mean": float(np.mean([r.metrics["mae"] for r in results])),
            "mae_std": float(np.std([r.metrics["mae"] for r in results])),
            "wape_mean": float(np.mean([r.metrics["wape"] for r in results])),
            "wape_std": float(np.std([r.metrics["wape"] for r in results])),
            "bias_mean": float(np.mean([r.metrics["bias"] for r in results])),
            "bias_std": float(np.std([r.metrics["bias"] for r in results])),
        },
    }

    (output_dir / f"demand_{run_name}_backtest.json").write_text(
        json.dumps(agg, indent=2, default=str), encoding="utf-8"
    )
    with (output_dir / f"demand_{run_name}_backtest_per_series.csv").open("w", newline="") as f:
        if per_series_rows:
            w = csv.DictWriter(f, fieldnames=list(per_series_rows[0].keys()))
            w.writeheader()
            w.writerows(per_series_rows)

    log.info("demand.backtest.done", extra={"n_folds": len(results), "output": str(output_dir)})
    return agg


def _main():
    p = argparse.ArgumentParser(description="Rolling-origin backtest of the demand model.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--output", default="reports/metrics")
    p.add_argument("--run-name", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-series", type=int, default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    result = run_backtest(
        args.checkpoint, args.features,
        output_dir=args.output, run_name=args.run_name,
        folds=args.folds, max_series=args.max_series, device=args.device,
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    _main()
