"""Evaluation: real-unit metrics + per-horizon breakdown + plots.

This module works on the same DatasetBundle produced by the training pipeline.
It never re-fits anything. Given a checkpoint directory, it produces:

    reports/metrics/demand_<run>_metrics.json
    reports/metrics/demand_<run>_per_horizon.csv
    reports/metrics/demand_<run>_per_series.csv
    reports/plots/demand_<run>_sample_forecasts.png
    reports/plots/demand_<run>_residual_hist.png
    reports/plots/demand_<run>_per_horizon_wape.png

All quantities are computed in real units (inverse-scaled) — no scaled error.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.logging import configure_logging, get_logger
from ml.demand.dataset import DemandDatasetBundle, build_demand_datasets
from ml.demand.model import DemandModelConfig, GlobalDemandLSTM

log = get_logger(__name__)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Metric functions — all in real units
# ---------------------------------------------------------------------------
def _mae(p, t):   return float(np.mean(np.abs(p - t)))
def _rmse(p, t):  return float(np.sqrt(np.mean((p - t) ** 2)))
def _wape(p, t):  return float(np.sum(np.abs(p - t)) / (np.sum(np.abs(t)) + _EPS))
def _smape(p, t): return float(np.mean(2 * np.abs(p - t) / (np.abs(p) + np.abs(t) + _EPS)))
def _bias(p, t):  return float(np.mean(p - t))
def _r2(p, t):
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2)) + _EPS
    return 1.0 - ss_res / ss_tot
def _mape(p, t):
    mask = np.abs(t) > _EPS
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((p[mask] - t[mask]) / t[mask])))


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Return point-forecast metrics. Inputs are shape (N, H)."""
    return {
        "mae": _mae(preds, targets),
        "rmse": _rmse(preds, targets),
        "wape": _wape(preds, targets),
        "smape": _smape(preds, targets),
        "mape": _mape(preds, targets),
        "bias": _bias(preds, targets),
        "r2": _r2(preds, targets),
        "n_samples": int(preds.shape[0]),
        "horizon": int(preds.shape[1]),
    }


# ---------------------------------------------------------------------------
# Inference over a dataset bundle
# ---------------------------------------------------------------------------
def _predict_dataset(
    model: GlobalDemandLSTM,
    loader: DataLoader,
    device: torch.device,
    scalers,
    quantiles: Optional[tuple],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_point_real, target_real, quantile_preds_or_none, series_keys).

    pred_point_real: (N, H)
    target_real:     (N, H)
    quantiles_real:  (N, H, Q) or None
    series_keys:     (N,) ints (series id) — via collate of series_sid
    """
    model.eval()
    preds_list, targets_list, quant_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            enc = batch["encoder"].to(device)
            dec = batch["decoder"].to(device)
            sc = batch["static_cat"].to(device)
            sn = batch["static_num"].to(device)
            pred = model(enc, dec, sc, sn)                 # (B, H) or (B, H, Q)

            if quantiles is None:
                pred_point = pred
                quant = None
            else:
                q_list = list(quantiles)
                med_idx = q_list.index(0.5) if 0.5 in q_list else len(q_list) // 2
                pred_point = pred[:, :, med_idx]
                quant = pred  # keep full quantile tensor

            # Invert scaling.
            pred_real = scalers.inverse_target(pred_point.detach().cpu().numpy())
            targets_real = batch["target_raw"].numpy()

            preds_list.append(pred_real)
            targets_list.append(targets_real)

            if quant is not None:
                quant_np = quant.detach().cpu().numpy()   # (B, H, Q) scaled
                # Inverse-scale each quantile row (target scaling applies to each).
                quant_real = scalers.inverse_target(quant_np)
                quant_list.append(quant_real)

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    quants = np.concatenate(quant_list, axis=0) if quant_list else None
    return preds, targets, quants, None


# ---------------------------------------------------------------------------
# Per-horizon breakdown
# ---------------------------------------------------------------------------
def per_horizon_metrics(preds: np.ndarray, targets: np.ndarray) -> List[Dict[str, float]]:
    H = preds.shape[1]
    rows = []
    for h in range(H):
        p = preds[:, h]
        t = targets[:, h]
        rows.append({
            "horizon": h + 1,
            "mae": _mae(p, t),
            "rmse": _rmse(p, t),
            "wape": _wape(p, t),
            "smape": _smape(p, t),
            "bias": _bias(p, t),
            "n": int(p.shape[0]),
        })
    return rows


# ---------------------------------------------------------------------------
# Per-series breakdown
# ---------------------------------------------------------------------------
def per_series_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    series_index_map: Dict[int, str],
) -> List[Dict[str, float]]:
    """series_index_map: row index → "sku|warehouse" label."""
    rows = []
    for i in range(preds.shape[0]):
        p = preds[i]
        t = targets[i]
        rows.append({
            "series": series_index_map.get(i, f"row_{i}"),
            "mae": _mae(p, t),
            "rmse": _rmse(p, t),
            "wape": _wape(p, t),
            "bias": _bias(p, t),
        })
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_sample_forecasts(
    preds: np.ndarray,
    targets: np.ndarray,
    quants: Optional[np.ndarray],
    out_path: Path,
    n_series: int = 6,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n_series, preds.shape[0])
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), squeeze=False)
    for i in range(n):
        ax = axes[i, 0]
        h = np.arange(1, preds.shape[1] + 1)
        ax.plot(h, targets[i], label="actual", marker="o", markersize=3, linewidth=1.2)
        ax.plot(h, preds[i], label="forecast", marker="x", markersize=3, linewidth=1.2)
        if quants is not None and quants.shape[-1] >= 3:
            q = sorted(range(quants.shape[-1]))
            lo_idx, hi_idx = q[0], q[-1]
            ax.fill_between(h, quants[i, :, lo_idx], quants[i, :, hi_idx],
                            alpha=0.2, label="interval")
        ax.set_title(f"Series {i}")
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_residual_hist(preds: np.ndarray, targets: np.ndarray, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resid = preds - targets
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(resid.flatten(), bins=60, alpha=0.85)
    ax.axvline(0.0, color="red", linewidth=1)
    ax.set_title("Demand forecast residuals (pred − actual)")
    ax.set_xlabel("residual (real units)")
    ax.set_ylabel("count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_per_horizon(per_h: List[Dict[str, float]], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = [r["horizon"] for r in per_h]
    wapes = [r["wape"] for r in per_h]
    maes = [r["mae"] for r in per_h]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(hs, wapes, marker="o", color="tab:blue", label="WAPE")
    ax1.set_xlabel("horizon (days ahead)")
    ax1.set_ylabel("WAPE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(hs, maes, marker="x", color="tab:orange", label="MAE")
    ax2.set_ylabel("MAE", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _load_checkpoint(checkpoint_dir: str | Path, device: torch.device):
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_path = checkpoint_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = checkpoint_dir / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint in {checkpoint_dir}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = DemandModelConfig(**ckpt["model_config"])
    model = GlobalDemandLSTM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, model_cfg, ckpt_path


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    features_dir: str | Path,
    *,
    output_dir: str | Path,
    run_name: str,
    batch_size: int = 512,
    device: str = "auto",
    max_series: Optional[int] = None,
) -> Dict:
    configure_logging()
    device_t = torch.device(
        device if device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata + scalers from checkpoint dir.
    from ml.demand.dataset import DemandScalers

    meta = json.loads((Path(checkpoint_dir) / "training_metadata.json").read_text())
    scalers = DemandScalers.load(Path(checkpoint_dir) / "scalers.npz")
    quantiles = tuple(meta["model_config"]["quantiles"]) if meta["model_config"]["quantiles"] else None

    # Rebuild datasets using the SAME split fractions the training used.
    # (These fractions are implicit in the training pipeline; we reproduce by
    # re-deriving from the metadata's split dates — the dataset builder honours
    # the same defaults so results are comparable. For strict invariance, the
    # training metadata records split dates; here we rebuild and then verify.)
    bundle = build_demand_datasets(
        features_dir,
        lookback=meta["train_config"]["lookback"],
        horizon=meta["train_config"]["horizon"],
        max_series=max_series,
    )

    # Verify split dates match training metadata.
    trained_splits = meta["split_dates"]
    rebuilt_splits = bundle.split_dates.to_dict()
    if trained_splits != rebuilt_splits:
        log.warning("demand.eval.split_mismatch", extra={
            "trained": trained_splits, "rebuilt": rebuilt_splits,
        })

    model, model_cfg, ckpt_path = _load_checkpoint(checkpoint_dir, device_t)

    def _loader(ds):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

    # Evaluate on all three splits.
    results: Dict[str, Dict] = {}
    for split_name, ds in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
        if len(ds) == 0:
            results[split_name] = {"n_samples": 0}
            continue
        preds, targets, quants, _ = _predict_dataset(model, _loader(ds), device_t, scalers, quantiles)
        metrics = compute_metrics(preds, targets)

        # Per-horizon + per-series CSVs (only for test split to keep output small).
        if split_name == "test":
            rows_h = per_horizon_metrics(preds, targets)
            import csv
            with (output_dir / f"demand_{run_name}_per_horizon.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_h[0].keys()))
                w.writeheader()
                w.writerows(rows_h)

            # Per-series: build label map from index id → label. We don't carry
            # the series key in the batch, so we use the series id from dataset index.
            series_map = {}
            for i, (sid, _t) in enumerate(ds.index):
                series_map[i] = f"series_{sid}"
            rows_s = per_series_metrics(preds, targets, series_map)
            with (output_dir / f"demand_{run_name}_per_series.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_s[0].keys()))
                w.writeheader()
                w.writerows(rows_s)

            # Plots.
            plots_dir = Path("reports/plots")
            _plot_sample_forecasts(
                preds, targets, quants,
                plots_dir / f"demand_{run_name}_sample_forecasts.png",
            )
            _plot_residual_hist(
                preds, targets,
                plots_dir / f"demand_{run_name}_residual_hist.png",
            )
            _plot_per_horizon(
                rows_h,
                plots_dir / f"demand_{run_name}_per_horizon_wape.png",
            )

        results[split_name] = metrics

    # Quantile coverage (if applicable).
    if quantiles is not None:
        _, _, quants_test, _ = _predict_dataset(
            model, _loader(bundle.test), device_t, scalers, quantiles,
        ) if len(bundle.test) > 0 else (None, None, None, None)
        if quants_test is not None:
            # We need targets too — recompute cleanly.
            preds_test, targets_test, quants_test, _ = _predict_dataset(
                model, _loader(bundle.test), device_t, scalers, quantiles,
            )
            coverage = {}
            for qi, q in enumerate(quantiles):
                lo = q < 0.5
                # Emit per-quantile empirical coverage.
                cov = float(np.mean(targets_test <= quants_test[:, :, qi]))
                coverage[f"q{q}"] = cov
            results["quantile_coverage"] = coverage

    # Persist JSON.
    payload = {
        "checkpoint": str(ckpt_path),
        "feature_version": meta["feature_version"],
        "run_name": run_name,
        "quantiles": list(quantiles) if quantiles else None,
        "metrics": results,
    }
    out_path = output_dir / f"demand_{run_name}_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("demand.eval.written", extra={"path": str(out_path)})

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main():
    p = argparse.ArgumentParser(description="Evaluate a trained demand checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--output", default="reports/metrics")
    p.add_argument("--run-name", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-series", type=int, default=None)
    args = p.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint, args.features,
        output_dir=args.output, run_name=args.run_name,
        batch_size=args.batch_size, device=args.device,
        max_series=args.max_series,
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    _main()