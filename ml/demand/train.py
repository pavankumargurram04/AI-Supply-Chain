"""Demand LSTM training loop.

Features:
- Deterministic seed (uses core.deterministic.seed_everything)
- Chronological train / val / test via ml.demand.dataset
- AdamW + cosine / ReduceLROnPlateau scheduler
- Gradient clipping
- Early stopping on validation loss
- Checkpoint on best validation loss
- Resume from checkpoint
- Full provenance saved to `training_metadata.json`
- Optional ModelRegistry row creation
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.config import get_settings
from core.deterministic import seed_everything
from core.logging import configure_logging, get_logger
from data_platform.features.feature_registry import FEATURE_VERSION
from ml.demand.dataset import (
    DemandDatasetBundle,
    build_demand_datasets,
)
from ml.demand.model import (
    DemandModelConfig,
    GlobalDemandLSTM,
    mse_loss,
    pinball_loss,
)

log = get_logger(__name__)


@dataclass
class TrainConfig:
    features_dir: str
    output_dir: str
    seed: int = 42
    lookback: int = 90
    horizon: int = 30
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    num_workers: int = 0
    device: str = "auto"
    max_series: Optional[int] = None
    quantiles: Optional[tuple] = None
    resume_from: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quantiles"] = list(self.quantiles) if self.quantiles else None
        return d


def _resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _make_loader(
    ds,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id: int) -> None:
        np.random.seed(seed + worker_id)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=g,
        worker_init_fn=_worker_init,
    )


def _compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: Optional[tuple],
) -> torch.Tensor:
    if quantiles is None:
        return mse_loss(pred, target)
    return pinball_loss(pred, target, quantiles)


def _run_epoch(
    model: GlobalDemandLSTM,
    loader: DataLoader,
    device: torch.device,
    optim: Optional[torch.optim.Optimizer],
    quantiles: Optional[tuple],
    grad_clip: float,
) -> float:
    training = optim is not None
    model.train(training)
    total_loss = 0.0
    n = 0

    for batch in loader:
        enc = batch["encoder"].to(device, non_blocking=True)
        dec = batch["decoder"].to(device, non_blocking=True)
        sc = batch["static_cat"].to(device, non_blocking=True)
        sn = batch["static_num"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            pred = model(enc, dec, sc, sn)
            loss = _compute_loss(pred, tgt, quantiles)

        if training:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        bs = enc.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        n += bs

    return total_loss / max(n, 1)


def _evaluate_full(
    model: GlobalDemandLSTM,
    loader: DataLoader,
    device: torch.device,
    scalers,
    quantiles: Optional[tuple],
) -> dict:
    """Compute loss + real-unit metrics on the given loader."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            enc = batch["encoder"].to(device)
            dec = batch["decoder"].to(device)
            sc = batch["static_cat"].to(device)
            sn = batch["static_num"].to(device)
            pred = model(enc, dec, sc, sn)

            if quantiles is None:
                pred_point = pred  # (B, H)
            else:
                # Use the median quantile.
                med_idx = list(quantiles).index(0.5) if 0.5 in quantiles else len(quantiles) // 2
                pred_point = pred[:, :, med_idx]

            # Invert scaling to real units.
            pred_np = pred_point.detach().cpu().numpy()
            pred_real = scalers.inverse_target(pred_np)
            tgt_real = batch["target_raw"].numpy()

            all_preds.append(pred_real)
            all_targets.append(tgt_real)

    if not all_preds:
        return {"n_samples": 0}

    P = np.concatenate(all_preds, axis=0)
    T = np.concatenate(all_targets, axis=0)

    eps = 1e-9
    mae = float(np.mean(np.abs(P - T)))
    rmse = float(np.sqrt(np.mean((P - T) ** 2)))
    denom = float(np.sum(np.abs(T))) + eps
    wape = float(np.sum(np.abs(P - T)) / denom)
    smape = float(np.mean(2 * np.abs(P - T) / (np.abs(P) + np.abs(T) + eps)))
    bias = float(np.mean(P - T))
    ss_res = float(np.sum((T - P) ** 2))
    ss_tot = float(np.sum((T - T.mean()) ** 2)) + eps
    r2 = 1.0 - ss_res / ss_tot

    return {
        "n_samples": int(P.shape[0]),
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "smape": smape,
        "bias": bias,
        "r2": r2,
    }


def train_demand_model(cfg: TrainConfig) -> dict:
    configure_logging()
    settings = get_settings()
    seed_everything(cfg.seed, deterministic=settings.DETERMINISTIC)

    device = _resolve_device(cfg.device)
    log.info("demand.train.start", extra={
        "device": str(device), "seed": cfg.seed, "output": cfg.output_dir,
        "feature_version": FEATURE_VERSION,
    })

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Datasets
    bundle: DemandDatasetBundle = build_demand_datasets(
        cfg.features_dir,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        max_series=cfg.max_series,
    )

    train_loader = _make_loader(
        bundle.train,
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, seed=cfg.seed,
    )
    val_loader = _make_loader(
        bundle.val,
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, seed=cfg.seed,
    )
    test_loader = _make_loader(
        bundle.test,
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, seed=cfg.seed,
    )

    # 2) Model
    model_cfg = DemandModelConfig(
        n_products=bundle.n_products,
        n_categories=bundle.n_categories,
        n_warehouses=bundle.n_warehouses,
        n_encoder_features=len(bundle.encoder_columns),
        n_decoder_features=len(bundle.decoder_columns),
        n_static_num=len(bundle.static_num_columns),
        horizon=cfg.horizon,
        quantiles=cfg.quantiles,
    )
    model = GlobalDemandLSTM(model_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("demand.train.model", extra={"params": n_params, **model_cfg.to_dict()})

    # 3) Optimiser + scheduler
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=2, min_lr=1e-5,
    )

    start_epoch = 0
    best_val = math.inf
    epochs_no_improve = 0
    history: list[dict] = []

    if cfg.resume_from:
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", math.inf)
        history = ckpt.get("history", [])
        log.info("demand.train.resumed", extra={"from": cfg.resume_from, "epoch": start_epoch})

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    # 4) Training loop
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_loss = _run_epoch(
            model, train_loader, device, optim, cfg.quantiles, cfg.grad_clip,
        )
        val_loss = _run_epoch(
            model, val_loader, device, None, cfg.quantiles, cfg.grad_clip,
        )
        elapsed = time.time() - t0
        scheduler.step(val_loss)

        log.info("demand.train.epoch", extra={
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "lr": float(optim.param_groups[0]["lr"]), "seconds": round(elapsed, 2),
        })

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(optim.param_groups[0]["lr"]),
            "seconds": elapsed,
        })

        # Save last checkpoint regardless.
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": best_val,
            "history": history,
            "model_config": model_cfg.to_dict(),
            "train_config": cfg.to_dict(),
        }, last_path)

        # Save best.
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "model_config": model_cfg.to_dict(),
                "val_loss": val_loss,
            }, best_path)
            log.info("demand.train.checkpoint", extra={"epoch": epoch, "val_loss": val_loss})
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.early_stop_patience:
            log.info("demand.train.early_stop", extra={"epoch": epoch})
            break

    # 5) Load best and evaluate
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    val_metrics = _evaluate_full(model, val_loader, device, bundle.scalers, cfg.quantiles)
    test_metrics = _evaluate_full(model, test_loader, device, bundle.scalers, cfg.quantiles)

    log.info("demand.train.eval", extra={"val": val_metrics, "test": test_metrics})

    # 6) Persist scalers + metadata
    bundle.scalers.save(out_dir / "scalers.npz")

    metadata = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION,
        "seed": cfg.seed,
        "device": str(device),
        "n_params": n_params,
        "model_config": model_cfg.to_dict(),
        "train_config": cfg.to_dict(),
        "split_dates": bundle.split_dates.to_dict(),
        "encoder_columns": bundle.encoder_columns,
        "decoder_columns": bundle.decoder_columns,
        "static_num_columns": bundle.static_num_columns,
        "n_products": bundle.n_products,
        "n_categories": bundle.n_categories,
        "n_warehouses": bundle.n_warehouses,
        "history": history,
        "best_val_loss": best_val,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
    }
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps({"val": val_metrics, "test": test_metrics}, indent=2, default=str),
        encoding="utf-8",
    )

    log.info("demand.train.done", extra={"best_val_loss": best_val, "output": str(out_dir)})
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> TrainConfig:
    import argparse

    p = argparse.ArgumentParser(description="Train the global demand LSTM.")
    p.add_argument("--features", required=True, help="Directory containing demand_features_*.parquet")
    p.add_argument("--output", required=True, help="Directory to write checkpoints and metadata")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-series", type=int, default=None)
    p.add_argument("--quantiles", type=str, default=None,
                   help="Comma-separated quantiles, e.g. 0.1,0.5,0.9")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    quantiles = None
    if args.quantiles:
        quantiles = tuple(float(x) for x in args.quantiles.split(","))

    return TrainConfig(
        features_dir=args.features,
        output_dir=args.output,
        seed=args.seed,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        device=args.device,
        max_series=args.max_series,
        quantiles=quantiles,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    meta = train_demand_model(cfg)
    print(json.dumps({"val": meta["val_metrics"], "test": meta["test_metrics"]}, indent=2))