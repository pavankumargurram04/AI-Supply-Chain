"""Build the (features, label) dataset for supplier risk.

Steps:
1. Load supplier_features_*.parquet (per (supplier_code, product_sku, date)).
2. Load supplier_events (the raw event log).
3. For each feature row at date t, label = 1 iff there is at least one
   disruptive event for that supplier_code in (t, t + horizon].
4. Drop rows whose future window is not fully observable (i.e. t + horizon
   exceeds the max event date in the dataset). This prevents training on
   partially-observed labels.
5. Split chronologically: train / val / test with global cut dates.
6. Return a bundle with the feature matrix, labels, split indices, and
   feature column list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.logging import get_logger
from data_platform.features.feature_registry import FEATURE_VERSION
from data_platform.splits.time_series_split import chronological_split

log = get_logger(__name__)


# Event types that constitute a "disruption" for risk modelling.
DISRUPTIVE_EVENT_TYPES: Tuple[str, ...] = (
    "delay",
    "quality_incident",
    "capacity_outage",
    "fill_rate_drop",
    "on_time_miss",
)


DATE_COLUMN = "date"
SUPPLIER_KEY = "supplier_code"
PRODUCT_KEY = "product_sku"


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
@dataclass
class SupplierDatasetBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: List[str]
    split_dates: Dict[str, str]
    feature_version: str
    horizon: int
    positive_rate_train: float
    positive_rate_val: float
    positive_rate_test: float
    # kept for downstream joins
    keys_train: pd.DataFrame = field(default_factory=pd.DataFrame)
    keys_val: pd.DataFrame = field(default_factory=pd.DataFrame)
    keys_test: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict:
        return {
            "n_train": int(len(self.X_train)),
            "n_val": int(len(self.X_val)),
            "n_test": int(len(self.X_test)),
            "n_features": len(self.feature_columns),
            "positive_rate_train": round(self.positive_rate_train, 6),
            "positive_rate_val": round(self.positive_rate_val, 6),
            "positive_rate_test": round(self.positive_rate_test, 6),
            "horizon": self.horizon,
            "feature_version": self.feature_version,
            "split_dates": self.split_dates,
        }


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------
def _resolve_feature_columns(df: pd.DataFrame) -> List[str]:
    """All `feat_sup_*` columns, excluding any that are target-derived.

    We never include a column that encodes the label window.
    """
    cols = [c for c in df.columns if c.startswith("feat_sup_")]
    # Sanity: never include anything that clearly represents a future window.
    forbidden = ("label", "target", "future")
    cols = [c for c in cols if not any(f in c.lower() for f in forbidden)]
    if not cols:
        raise ValueError(
            "No feat_sup_* columns found. Did build_features.py run successfully?"
        )
    return cols


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------
def _build_supplier_event_index(events: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return {supplier_code: sorted array of disruptive event timestamps (ns int)}."""
    if events.empty:
        return {}
    ev = events.copy()
    ev["occurred_at"] = pd.to_datetime(ev["occurred_at"])
    ev = ev[ev["event_type"].isin(DISRUPTIVE_EVENT_TYPES)]
    if ev.empty:
        return {}
    groups = ev.groupby(SUPPLIER_KEY)["occurred_at"].apply(
        lambda s: np.sort(s.astype("int64").to_numpy())
    )
    return groups.to_dict()


def _label_rows(
    features: pd.DataFrame,
    event_index: Dict[str, np.ndarray],
    *,
    horizon_days: int,
    max_observable_date: pd.Timestamp,
) -> pd.Series:
    """Return a boolean Series aligned to `features.index`."""
    feature_dates = pd.to_datetime(features[DATE_COLUMN]).astype("int64").to_numpy()
    supplier_codes = features[SUPPLIER_KEY].to_numpy()
    horizon_ns = horizon_days * 86_400_000_000_000

    labels = np.zeros(len(features), dtype=np.int8)
    observable = np.zeros(len(features), dtype=bool)

    for i in range(len(features)):
        t = feature_dates[i]
        end = t + horizon_ns
        code = supplier_codes[i]
        arr = event_index.get(code)

        # Row is observable only if the full window fits within the dataset.
        if end <= int(max_observable_date.value):
            observable[i] = True
        if arr is None:
            continue
        # Events strictly after t and <= t+H.
        left = np.searchsorted(arr, t, side="right")
        right = np.searchsorted(arr, end, side="right")
        if right > left:
            labels[i] = 1

    out = pd.Series(labels, index=features.index, name="label")
    out.loc[~observable] = -1  # mark as not observable
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _locate_supplier_features(features_dir: Path) -> Path:
    candidates = sorted(features_dir.glob("supplier_features_*.parquet"))
    if not candidates:
        candidates = sorted((features_dir / "features").glob("supplier_features_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No supplier_features_*.parquet under {features_dir}")
    return candidates[-1]


def build_supplier_dataset(
    features_dir: str | Path,
    *,
    raw_dir: Optional[str | Path] = None,
    horizon_days: int = 30,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    max_rows: Optional[int] = None,
) -> SupplierDatasetBundle:
    """Build train / val / test for supplier risk.

    Args:
        features_dir: directory containing supplier_features_*.parquet.
        raw_dir: directory containing supplier_events Parquet (defaults to
            sibling of features_dir named "../raw/<mode>" if not provided,
            otherwise features_dir itself is searched).
        horizon_days: the future window over which we compute labels.
        val_fraction, test_fraction: chronological split fractions.
        max_rows: optional cap (for CI / smoke runs).
    """
    features_dir = Path(features_dir)
    features_path = _locate_supplier_features(features_dir)
    log.info("supplier.dataset.loading_features", extra={"path": str(features_path)})
    features = pd.read_parquet(features_path)

    if features.empty:
        raise ValueError(f"Empty supplier feature file: {features_path}")

    # Locate events.
    events_path: Optional[Path] = None
    if raw_dir is not None:
        cand = Path(raw_dir) / "supplier_events"
        if cand.is_dir():
            events_path = cand
        else:
            alt = Path(raw_dir) / "supplier_events" / "data.parquet"
            if alt.exists():
                events_path = alt
    if events_path is None:
        # Try sibling `raw` directory next to curated.
        guess = features_dir.parent.parent / "raw"
        if guess.exists():
            for sub in guess.iterdir():
                ev = sub / "supplier_events"
                if ev.is_dir():
                    events_path = ev
                    break
    if events_path is None:
        raise FileNotFoundError(
            "Could not locate supplier_events Parquet. Pass --raw explicitly."
        )

    log.info("supplier.dataset.loading_events", extra={"path": str(events_path)})
    events = pd.read_parquet(events_path)
    if events.empty:
        raise ValueError(f"Empty supplier events file: {events_path}")

    # Build label.
    max_event_date = pd.to_datetime(events["occurred_at"]).max().normalize()
    event_index = _build_supplier_event_index(events)

    features = features.sort_values([SUPPLIER_KEY, PRODUCT_KEY, DATE_COLUMN]).reset_index(drop=True)
    labels = _label_rows(
        features, event_index,
        horizon_days=horizon_days,
        max_observable_date=max_event_date,
    )
    features["__label__"] = labels
    features = features[features["__label__"] != -1].reset_index(drop=True)
    log.info("supplier.dataset.labeled", extra={
        "rows": len(features),
        "positive_rate": float(features["__label__"].mean()),
        "horizon_days": horizon_days,
    })

    if max_rows is not None:
        features = features.head(max_rows).reset_index(drop=True)

    # Resolve features.
    feature_columns = _resolve_feature_columns(features)

    # Chronological split — global cut dates from `date`.
    train_df, val_df, test_df, split_dates = chronological_split(
        features, date_col=DATE_COLUMN,
        val_fraction=val_fraction, test_fraction=test_fraction,
    )

    def _xy(df: pd.DataFrame):
        X = df[feature_columns].copy()
        # Coerce categorical strings to numeric codes fit on train only.
        # We do this AFTER the split so no category from val/test leaks in.
        y = df["__label__"].astype(int)
        keys = df[[SUPPLIER_KEY, PRODUCT_KEY, DATE_COLUMN]].reset_index(drop=True)
        return X, y, keys

    X_train, y_train, keys_train = _xy(train_df)
    X_val, y_val, keys_val = _xy(val_df)
    X_test, y_test, keys_test = _xy(test_df)

    # Fit category codes on train, apply to val/test with unknown → -1.
    cat_cols = [c for c in feature_columns if X_train[c].dtype == object]
    for c in cat_cols:
        uniques = sorted(X_train[c].dropna().unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        X_train[c] = X_train[c].map(mapping).astype("int32")
        X_val[c] = X_val[c].map(mapping).fillna(-1).astype("int32")
        X_test[c] = X_test[c].map(mapping).fillna(-1).astype("int32")

    # Fill remaining NaNs (numeric) with 0 — safe for tree models and
    # consistent across splits.
    for X in (X_train, X_val, X_test):
        for c in X.columns:
            if X[c].isna().any():
                X[c] = X[c].fillna(0.0)

    bundle = SupplierDatasetBundle(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_columns=feature_columns,
        split_dates=split_dates.to_dict(),
        feature_version=FEATURE_VERSION,
        horizon=horizon_days,
        positive_rate_train=float(y_train.mean()) if len(y_train) else 0.0,
        positive_rate_val=float(y_val.mean()) if len(y_val) else 0.0,
        positive_rate_test=float(y_test.mean()) if len(y_test) else 0.0,
        keys_train=keys_train, keys_val=keys_val, keys_test=keys_test,
    )
    log.info("supplier.dataset.built", extra=bundle.summary())
    return bundle