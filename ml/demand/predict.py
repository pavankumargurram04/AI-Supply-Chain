"""Single-series demand inference with full provenance.

Loads a checkpoint, resolves the last `lookback` days of encoder features
and the next `horizon` days of decoder calendar features, and produces:

    - point forecast (real units)
    - uncertainty intervals (if quantile head or MC-dropout enabled)
    - model/data/feature version tags for downstream provenance

The function is deterministic: same input rows + same checkpoint → same
output bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from core.logging import get_logger
from data_platform.features.feature_registry import FEATURE_VERSION
from ml.demand.dataset import (
    DECODER_PREFIXES,
    ENCODER_PREFIXES,
    SERIES_KEYS,
    STATIC_NUM_COLUMNS,
    TARGET_COLUMN,
    DATE_COLUMN,
    DemandScalers,
)
from ml.demand.model import DemandModelConfig, GlobalDemandLSTM

log = get_logger(__name__)


@dataclass
class DemandForecast:
    product_sku: str
    warehouse_code: str
    horizon: int
    origin_date: str
    forecast_dates: List[str]
    point: List[float]
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None
    quantiles: Optional[dict] = None
    model_version: str = ""
    feature_version: str = FEATURE_VERSION
    data_version: Optional[str] = None
    seed: int = 42
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "product_sku": self.product_sku,
            "warehouse_code": self.warehouse_code,
            "horizon": self.horizon,
            "origin_date": self.origin_date,
            "forecast_dates": self.forecast_dates,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "quantiles": self.quantiles,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "data_version": self.data_version,
            "seed": self.seed,
            "extras": self.extras,
        }


class DemandPredictor:
    """Loads a checkpoint once and can be called repeatedly for different series."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "auto") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(
            device if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Metadata + scalers.
        meta_path = self.checkpoint_dir / "training_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing training metadata: {meta_path}")
        self.meta = json.loads(meta_path.read_text())
        self.scalers = DemandScalers.load(self.checkpoint_dir / "scalers.npz")

        # Model.
        ckpt_path = self.checkpoint_dir / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = self.checkpoint_dir / "last.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model_cfg = DemandModelConfig(**ckpt["model_config"])
        self.model = GlobalDemandLSTM(self.model_cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.quantiles = (
            tuple(self.meta["model_config"]["quantiles"])
            if self.meta["model_config"].get("quantiles") else None
        )
        self.encoder_columns = list(self.meta["encoder_columns"])
        self.decoder_columns = list(self.meta["decoder_columns"])
        self.static_num_columns = list(self.meta["static_num_columns"])
        self.lookback = int(self.meta["train_config"]["lookback"])
        self.horizon = int(self.meta["train_config"]["horizon"])
        self.model_version = self.meta.get("model_version", "unnamed")

    # -- public API -----------------------------------------------------
    def predict(
        self,
        feature_df: pd.DataFrame,
        *,
        product_sku: str,
        warehouse_code: str,
        origin_date: Optional[pd.Timestamp] = None,
        mc_dropout_samples: int = 0,
    ) -> DemandForecast:
        """Forecast for one series using a slice of the feature dataframe.

        `feature_df` must contain the same columns as the demand feature table
        (typically loaded from `datasets/curated/<mode>/features/demand_features_*.parquet`).
        """
        df = feature_df
        mask = (df["product_sku"] == product_sku) & (df["warehouse_code"] == warehouse_code)
        series = df[mask].sort_values(DATE_COLUMN).reset_index(drop=True)
        if series.empty:
            raise ValueError(f"No rows for series ({product_sku}, {warehouse_code}).")

        if origin_date is None:
            # Use the latest complete window.
            origin_idx = len(series) - self.horizon - 1
        else:
            idxs = series.index[series[DATE_COLUMN] == pd.Timestamp(origin_date)].tolist()
            if not idxs:
                raise ValueError(f"origin_date {origin_date} not found in series.")
            origin_idx = idxs[0]

        if origin_idx < self.lookback - 1:
            raise ValueError("Not enough history for the requested origin.")

        enc_slice = series.iloc[origin_idx - self.lookback + 1 : origin_idx + 1]
        dec_slice = series.iloc[origin_idx + 1 : origin_idx + 1 + self.horizon]
        if len(dec_slice) < self.horizon:
            raise ValueError("Not enough future rows (calendar features) in the frame.")

        enc_raw = enc_slice[self.encoder_columns].to_numpy(dtype=np.float32)
        dec_raw = dec_slice[self.decoder_columns].to_numpy(dtype=np.float32)
        enc_scaled = self.scalers.transform_encoder(np.nan_to_num(enc_raw, nan=0.0))
        dec_scaled = np.nan_to_num(dec_raw, nan=0.0)

        # Static inputs — read from the encoder slice's first row.
        first = enc_slice.iloc[0]
        # Embedding indices were baked into the training set. We use index 0
        # (the "unknown" slot) for anything the checkpoint did not see; the
        # checkpoint's training metadata documents which SKUs / warehouses
        # were in the training vocabulary.
        prod_idx = int(self.meta.get("product_index_map", {}).get(product_sku, 0))
        cat_idx = int(self.meta.get("category_index_map", {}).get(str(first.get("category_code", "")), 0))
        wh_idx = int(self.meta.get("warehouse_index_map", {}).get(warehouse_code, 0))
        static_cat = np.array([prod_idx, cat_idx, wh_idx], dtype=np.int64)
        static_num = np.array([
            float(first.get("unit_cost", 0.0)),
            float(first.get("unit_price", 0.0)),
            float(first.get("base_demand", 0.0)),
            float(first.get("lead_time_days", 0.0)),
        ], dtype=np.float32)

        enc_t = torch.from_numpy(enc_scaled[None, :, :]).to(self.device)
        dec_t = torch.from_numpy(dec_scaled[None, :, :]).to(self.device)
        sc_t = torch.from_numpy(static_cat[None, :]).to(self.device)
        sn_t = torch.from_numpy(static_num[None, :]).to(self.device)

        point_scaled = self._forward_point(enc_t, dec_t, sc_t, sn_t)
        point_real = self.scalers.inverse_target(point_scaled)[0]

        lower = upper = None
        quant_dict = None

        if self.quantiles is not None:
            with torch.no_grad():
                pred = self.model(enc_t, dec_t, sc_t, sn_t)      # (1, H, Q)
            pred_real = self.scalers.inverse_target(pred.detach().cpu().numpy())[0]  # (H, Q)
            quant_dict = {f"q{q}": pred_real[:, i].tolist() for i, q in enumerate(self.quantiles)}
            lo_idx = list(self.quantiles).index(min(self.quantiles))
            hi_idx = list(self.quantiles).index(max(self.quantiles))
            lower = pred_real[:, lo_idx].tolist()
            upper = pred_real[:, hi_idx].tolist()

        if mc_dropout_samples > 0:
            mc_preds = self._mc_dropout(enc_t, dec_t, sc_t, sn_t, mc_dropout_samples)
            lower = np.percentile(mc_preds, 10, axis=0).tolist() if lower is None else lower
            upper = np.percentile(mc_preds, 90, axis=0).tolist() if upper is None else upper

        forecast_dates = [pd.Timestamp(d).date().isoformat() for d in dec_slice[DATE_COLUMN].tolist()]
        origin_date_str = pd.Timestamp(enc_slice[DATE_COLUMN].iloc[-1]).date().isoformat()

        return DemandForecast(
            product_sku=product_sku,
            warehouse_code=warehouse_code,
            horizon=self.horizon,
            origin_date=origin_date_str,
            forecast_dates=forecast_dates,
            point=[float(x) for x in point_real],
            lower=lower,
            upper=upper,
            quantiles=quant_dict,
            model_version=self.model_version,
            feature_version=self.meta.get("feature_version", FEATURE_VERSION),
            data_version=self.meta.get("data_version"),
            seed=int(self.meta.get("seed", 42)),
            extras={
                "checkpoint_dir": str(self.checkpoint_dir),
                "mc_dropout_samples": mc_dropout_samples,
            },
        )

    # -- internals ------------------------------------------------------
    def _forward_point(self, enc_t, dec_t, sc_t, sn_t) -> np.ndarray:
        with torch.no_grad():
            pred = self.model(enc_t, dec_t, sc_t, sn_t)
            if self.quantiles is not None:
                q_list = list(self.quantiles)
                med_idx = q_list.index(0.5) if 0.5 in q_list else len(q_list) // 2
                pred = pred[:, :, med_idx]
            return pred.detach().cpu().numpy()

    def _mc_dropout(self, enc_t, dec_t, sc_t, sn_t, samples: int) -> np.ndarray:
        # Turn dropout layers back on for MC sampling.
        self.model.train()
        preds = []
        try:
            with torch.no_grad():
                for _ in range(samples):
                    pred = self.model(enc_t, dec_t, sc_t, sn_t)
                    if self.quantiles is not None:
                        q_list = list(self.quantiles)
                        med_idx = q_list.index(0.5) if 0.5 in q_list else len(q_list) // 2
                        pred = pred[:, :, med_idx]
                    preds.append(self.scalers.inverse_target(pred.detach().cpu().numpy())[0])
        finally:
            self.model.eval()
        return np.stack(preds, axis=0)  # (S, H)