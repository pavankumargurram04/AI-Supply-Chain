"""Demand sequence builder and torch Dataset.

Responsibilities:
1. Load the demand feature table produced by Phase 5.
2. Split chronologically into train / val / test (global cut dates).
3. Fit the target scaler and encoder-feature scaler ONLY on the training split.
4. Expose windowed samples: encoder input [t-L+1..t], decoder calendar input
   [t+1..t+H], target [t+1..t+H].

Design guarantees:
- No feature at index <= t uses a target at index <= t.
- The scaler never sees validation or test rows.
- Static embedding indices are built from training data only; unknown ids at
  val/test time are mapped to an "unknown" slot (index 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.logging import get_logger
from data_platform.splits.leakage_detector import assert_no_leakage
from data_platform.splits.time_series_split import (
    SplitDates,
    chronological_split,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

# Columns that carry historical demand information. Only present in the encoder.
ENCODER_PREFIXES: Tuple[str, ...] = (
    "feat_demand_lag_",
    "feat_demand_roll_",
    "feat_demand_expand_",
    "feat_promo_count_",
    "feat_holiday_count_",
    "feat_censored_count_",
)

# Columns that describe the calendar at the *target* date. Safe to use in the
# decoder because they are known in advance (we forecast calendar, not demand).
DECODER_PREFIXES: Tuple[str, ...] = (
    "feat_dow",         # feat_dow, feat_dow_sin, feat_dow_cos
    "feat_dom",
    "feat_month",       # feat_month, feat_month_sin, feat_month_cos
    "feat_quarter",
    "feat_year",
    "feat_week",
    "feat_day_of_year",
    "feat_doy_",        # feat_doy_sin, feat_doy_cos
    "feat_is_",
    "feat_days_to_holiday",
)

STATIC_CAT_COLUMNS: Tuple[str, ...] = ("product_sku", "category_code", "warehouse_code")
STATIC_NUM_COLUMNS: Tuple[str, ...] = (
    "unit_cost",
    "unit_price",
    "base_demand",
    "lead_time_days",
)

TARGET_COLUMN = "quantity"
DATE_COLUMN = "date"
SERIES_KEYS: Tuple[str, ...] = ("product_sku", "warehouse_code")


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------
@dataclass
class DemandScalers:
    """Scalers fit only on the training split.

    Attributes:
        enc_mean, enc_std: per-column mean/std for encoder features.
        target_mean, target_std: mean/std of log1p(target) on train.
        encoder_columns, decoder_columns, static_num_columns: ordered feature names.
    """

    enc_mean: np.ndarray
    enc_std: np.ndarray
    target_mean: float
    target_std: float
    encoder_columns: List[str]
    decoder_columns: List[str]
    static_num_columns: List[str]

    def transform_encoder(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.enc_mean) / self.enc_std

    def transform_target(self, arr: np.ndarray) -> np.ndarray:
        return (np.log1p(arr) - self.target_mean) / self.target_std

    def inverse_target(self, arr: np.ndarray) -> np.ndarray:
        return np.expm1(arr * self.target_std + self.target_mean)

    def to_dict(self) -> dict:
        return {
            "encoder_columns": list(self.encoder_columns),
            "decoder_columns": list(self.decoder_columns),
            "static_num_columns": list(self.static_num_columns),
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            enc_mean=self.enc_mean,
            enc_std=self.enc_std,
            target_mean=np.array([self.target_mean]),
            target_std=np.array([self.target_std]),
            encoder_columns=np.array(self.encoder_columns),
            decoder_columns=np.array(self.decoder_columns),
            static_num_columns=np.array(self.static_num_columns),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DemandScalers":
        data = np.load(path, allow_pickle=True)
        return cls(
            enc_mean=data["enc_mean"],
            enc_std=data["enc_std"],
            target_mean=float(data["target_mean"][0]),
            target_std=float(data["target_std"][0]),
            encoder_columns=list(data["encoder_columns"]),
            decoder_columns=list(data["decoder_columns"]),
            static_num_columns=list(data["static_num_columns"]),
        )


# ---------------------------------------------------------------------------
# Dataset bundle (the materialised window index and per-series arrays)
# ---------------------------------------------------------------------------
@dataclass
class SeriesData:
    encoder: np.ndarray      # shape (T, n_enc), float32
    decoder: np.ndarray      # shape (T, n_dec), float32
    target: np.ndarray       # shape (T,), raw demand, float32
    target_scaled: np.ndarray  # shape (T,), scaled target, float32
    static_cat: np.ndarray   # shape (3,), int64: [prod_idx, cat_idx, wh_idx]
    static_num: np.ndarray   # shape (n_static_num,), float32
    valid_mask: np.ndarray   # shape (T,), bool — False where NaN present in encoder


@dataclass
class DemandDatasetBundle:
    train: "WindowedDemandDataset"
    val: "WindowedDemandDataset"
    test: "WindowedDemandDataset"
    scalers: DemandScalers
    split_dates: SplitDates
    n_products: int
    n_categories: int
    n_warehouses: int
    encoder_columns: List[str]
    decoder_columns: List[str]
    static_num_columns: List[str]
    # Kept for provenance / debugging.
    series_meta: Dict[Tuple[str, str], dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Windowed Dataset
# ---------------------------------------------------------------------------
class WindowedDemandDataset(Dataset):
    """A lazily-indexed windowed view over the series arrays.

    Each sample is a tuple (encoder, decoder, static_cat, static_num, target).
    We store one entry per (series_id, t) in `index`, and slice on demand.
    """

    def __init__(
        self,
        series: List[SeriesData],
        index: List[Tuple[int, int]],
        lookback: int,
        horizon: int,
    ) -> None:
        self.series = series
        self.index = index
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sid, t = self.index[i]
        s = self.series[sid]

        enc = s.encoder[t - self.lookback + 1 : t + 1]           # (L, n_enc)
        dec = s.decoder[t + 1 : t + 1 + self.horizon]            # (H, n_dec)
        tgt = s.target_scaled[t + 1 : t + 1 + self.horizon]      # (H,)

        return {
            "encoder": torch.from_numpy(enc.astype(np.float32, copy=False)),
            "decoder": torch.from_numpy(dec.astype(np.float32, copy=False)),
            "static_cat": torch.from_numpy(s.static_cat.astype(np.int64, copy=False)),
            "static_num": torch.from_numpy(s.static_num.astype(np.float32, copy=False)),
            "target": torch.from_numpy(tgt.astype(np.float32, copy=False)),
            # Keep the raw target too for metric computation at eval time.
            "target_raw": torch.from_numpy(
                s.target[t + 1 : t + 1 + self.horizon].astype(np.float32, copy=False)
            ),
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def _resolve_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    enc_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in ENCODER_PREFIXES)
    ]
    dec_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in DECODER_PREFIXES)
    ]
    if not enc_cols:
        raise ValueError(
            "No encoder columns found. Did build_features.py run successfully?"
        )
    if not dec_cols:
        raise ValueError("No decoder columns found.")
    return enc_cols, dec_cols


def _fit_scalers(
    train_df: pd.DataFrame,
    encoder_columns: List[str],
) -> DemandScalers:
    enc = train_df[encoder_columns].to_numpy(dtype=np.float64, copy=False)
    enc_mean = np.nanmean(enc, axis=0)
    enc_std = np.nanstd(enc, axis=0)
    # Guard against zero-variance columns (constant features).
    enc_std = np.where(enc_std < 1e-8, 1.0, enc_std)

    target_raw = train_df[TARGET_COLUMN].to_numpy(dtype=np.float64, copy=False)
    log_target = np.log1p(target_raw)
    t_mean = float(np.mean(log_target))
    t_std = float(np.std(log_target))
    if t_std < 1e-8:
        t_std = 1.0

    return DemandScalers(
        enc_mean=enc_mean.astype(np.float32),
        enc_std=enc_std.astype(np.float32),
        target_mean=t_mean,
        target_std=t_std,
        encoder_columns=list(encoder_columns),
        decoder_columns=[],  # filled by caller
        static_num_columns=list(STATIC_NUM_COLUMNS),
    )


def _build_series_arrays(
    df: pd.DataFrame,
    *,
    encoder_columns: List[str],
    decoder_columns: List[str],
    scalers: DemandScalers,
    product_to_idx: Dict[str, int],
    category_to_idx: Dict[str, int],
    warehouse_to_idx: Dict[str, int],
) -> Dict[Tuple[str, str], SeriesData]:
    """Group by (product, warehouse), sort by date, produce per-series arrays."""
    df = df.sort_values(list(SERIES_KEYS) + [DATE_COLUMN])

    series_arrays: Dict[Tuple[str, str], SeriesData] = {}

    for (sku, wh), g in df.groupby(list(SERIES_KEYS), sort=False):
        enc_raw = g[encoder_columns].to_numpy(dtype=np.float32, copy=True)
        dec_raw = g[decoder_columns].to_numpy(dtype=np.float32, copy=True)
        tgt_raw = g[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True)

        # NaN guard: any row with NaN in the encoder is excluded from sampling.
        # We keep the row in the array (so indices align) but mark it invalid.
        valid = ~np.isnan(enc_raw).any(axis=1) & ~np.isnan(dec_raw).any(axis=1)

        # Replace NaN with 0 so numpy ops don't poison downstream.
        enc_filled = np.nan_to_num(enc_raw, nan=0.0)
        dec_filled = np.nan_to_num(dec_raw, nan=0.0)

        enc_scaled = scalers.transform_encoder(enc_filled)
        tgt_scaled = scalers.transform_target(tgt_raw)

        product_idx = product_to_idx.get(sku, 0)
        category_idx = category_to_idx.get(str(g["category_code"].iloc[0]), 0)
        warehouse_idx = warehouse_to_idx.get(wh, 0)

        static_num = np.array([
            float(g["unit_cost"].iloc[0]),
            float(g["unit_price"].iloc[0]),
            float(g["base_demand"].iloc[0]),
            float(g["lead_time_days"].iloc[0]),
        ], dtype=np.float32)

        series_arrays[(sku, wh)] = SeriesData(
            encoder=enc_scaled.astype(np.float32),
            decoder=dec_filled.astype(np.float32),
            target=tgt_raw.astype(np.float32),
            target_scaled=tgt_scaled.astype(np.float32),
            static_cat=np.array([product_idx, category_idx, warehouse_idx], dtype=np.int64),
            static_num=static_num,
            valid_mask=valid,
        )

    return series_arrays


def _collect_index(
    series: List[SeriesData],
    *,
    lookback: int,
    horizon: int,
    series_ids: Sequence[Tuple[str, str]],
    key_to_id: Dict[Tuple[str, str], int],
) -> List[Tuple[int, int]]:
    """Build the (series_id, t) index for a subset of series."""
    idx: List[Tuple[int, int]] = []
    for key in series_ids:
        sid = key_to_id.get(key)
        if sid is None:
            continue
        s = series[sid]
        T = len(s.target)
        # Need L rows before t and H rows after t.
        for t in range(lookback - 1, T - horizon):
            # Skip windows whose encoder or decoder contains invalid rows.
            if not s.valid_mask[t - lookback + 1 : t + 1].all():
                continue
            if not s.valid_mask[t + 1 : t + 1 + horizon].all():
                continue
            idx.append((sid, t))
    return idx


def build_demand_datasets(
    features_dir: str | Path,
    *,
    lookback: int = 90,
    horizon: int = 30,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    max_series: Optional[int] = None,
    feature_filename: Optional[str] = None,
) -> DemandDatasetBundle:
    """Build train / val / test datasets from a curated feature directory.

    Args:
        features_dir: directory containing `demand_features_<version>.parquet`
            (or a directory under `datasets/curated/<mode>/features/`).
        lookback: encoder window length in days.
        horizon: forecast horizon in days.
        val_fraction / test_fraction: chronological split fractions.
        max_series: optional cap on the number of series (dev / CI speed).
        feature_filename: override the resolved feature file name.
    """
    features_dir = Path(features_dir)
    if feature_filename is None:
        candidates = sorted(features_dir.glob("demand_features_*.parquet"))
        if not candidates:
            # Allow passing the parent dir directly (datasets/curated/<mode>).
            candidates = sorted((features_dir / "features").glob("demand_features_*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"No demand_features_*.parquet found under {features_dir}"
            )
        feature_path = candidates[-1]
    else:
        feature_path = features_dir / feature_filename

    # Research feature tables are too large for a full pandas load on an
    # 8GB-class workstation. Use the streaming builder instead.
    if "research" in feature_path.parts:
        from ml.demand.research_dataset import (
            build_demand_datasets_research,
        )

        return build_demand_datasets_research(
            feature_path,
            lookback=lookback,
            horizon=horizon,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            max_series=max_series,
        )
    log.info("demand.dataset.loading", extra={"path": str(feature_path)})
    df = pd.read_parquet(feature_path)
    if df.empty:
        raise ValueError(f"Feature file is empty: {feature_path}")

    # Filter to active series only.
    df = df.sort_values(list(SERIES_KEYS) + [DATE_COLUMN]).reset_index(drop=True)

    if max_series is not None:
        keep = (
            df[list(SERIES_KEYS)]
            .drop_duplicates()
            .head(max_series)
        )
        df = df.merge(keep, on=list(SERIES_KEYS), how="inner").reset_index(drop=True)

    # 1) Split chronologically.
    train_df, val_df, test_df, split_dates = chronological_split(
        df, date_col=DATE_COLUMN,
        val_fraction=val_fraction, test_fraction=test_fraction,
    )
    assert_no_leakage(train_df, val_df, test_df, split_dates, date_col=DATE_COLUMN)

    log.info(
        "demand.dataset.split",
        extra={
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "val_start": split_dates.val_start.isoformat(),
            "test_start": split_dates.test_start.isoformat(),
        },
    )

    encoder_columns, decoder_columns = _resolve_columns(df)

    # 2) Fit scalers on train.
    scalers = _fit_scalers(train_df, encoder_columns)
    scalers.decoder_columns = decoder_columns

    # 3) Build embedding indices from TRAIN only.
    product_to_idx = {sku: i + 1 for i, sku in enumerate(sorted(train_df["product_sku"].unique()))}
    category_to_idx = {c: i + 1 for i, c in enumerate(sorted(train_df["category_code"].dropna().unique()))}
    warehouse_to_idx = {w: i + 1 for i, w in enumerate(sorted(train_df["warehouse_code"].unique()))}
    # Index 0 is reserved for "unknown" (unseen at training time).

    # 4) Build per-series arrays using the full dataframe (so windows that
    #    cross the split boundary still have contiguous history — this is
    #    correct for time-series: we may use past history before val_start
    #    to predict at val_start). The windows themselves are split by their
    #    *forecast origin* index t.

    series_arrays = _build_series_arrays(
        df,
        encoder_columns=encoder_columns,
        decoder_columns=decoder_columns,
        scalers=scalers,
        product_to_idx=product_to_idx,
        category_to_idx=category_to_idx,
        warehouse_to_idx=warehouse_to_idx,
    )

    series_list = list(series_arrays.values())
    key_to_id = {k: i for i, k in enumerate(series_arrays.keys())}

    # 5) Compute split membership by date.
    train_dates = set(pd.to_datetime(train_df[DATE_COLUMN]).dt.normalize().unique())
    val_dates = set(pd.to_datetime(val_df[DATE_COLUMN]).dt.normalize().unique())
    test_dates = set(pd.to_datetime(test_df[DATE_COLUMN]).dt.normalize().unique())

    # We need to look up the date of the *forecast origin* (index t).
    # Build a per-series date array.
    date_arrays: Dict[Tuple[str, str], np.ndarray] = {}
    for (sku, wh), g in df.groupby(list(SERIES_KEYS), sort=False):
        g = g.sort_values(DATE_COLUMN)
        date_arrays[(sku, wh)] = pd.to_datetime(g[DATE_COLUMN]).to_numpy()

    train_idx: List[Tuple[int, int]] = []
    val_idx: List[Tuple[int, int]] = []
    test_idx: List[Tuple[int, int]] = []

    for key, sid in key_to_id.items():
        s = series_list[sid]
        dates = date_arrays[key]
        T = len(s.target)
        for t in range(lookback - 1, T - horizon):
            if not s.valid_mask[t - lookback + 1 : t + 1].all():
                continue
            if not s.valid_mask[t + 1 : t + 1 + horizon].all():
                continue
            origin_date = pd.Timestamp(dates[t]).normalize()
            # Classify by the forecast origin.
            if origin_date in train_dates:
                train_idx.append((sid, t))
            elif origin_date in val_dates:
                val_idx.append((sid, t))
            elif origin_date in test_dates:
                test_idx.append((sid, t))

    log.info(
        "demand.dataset.windows",
        extra={"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    )

    if not train_idx:
        raise ValueError(
            "No training windows produced. Reduce lookback/horizon or expand the dataset."
        )
    if not val_idx:
        log.warning("demand.dataset.no_val_windows")
    if not test_idx:
        log.warning("demand.dataset.no_test_windows")

    train_ds = WindowedDemandDataset(series_list, train_idx, lookback, horizon)
    val_ds = WindowedDemandDataset(series_list, val_idx, lookback, horizon)
    test_ds = WindowedDemandDataset(series_list, test_idx, lookback, horizon)

    return DemandDatasetBundle(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        scalers=scalers,
        split_dates=split_dates,
        n_products=len(product_to_idx) + 1,   # +1 for unknown slot
        n_categories=len(category_to_idx) + 1,
        n_warehouses=len(warehouse_to_idx) + 1,
        encoder_columns=encoder_columns,
        decoder_columns=decoder_columns,
        static_num_columns=list(STATIC_NUM_COLUMNS),
        series_meta={k: {"n_rows": len(v.target)} for k, v in series_arrays.items()},
    )
