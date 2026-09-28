"""Residual extraction.

Residuals are the foundation of the probabilistic stockout simulation
downstream. We persist them per (product, warehouse, horizon) so the
simulation can bootstrap them without re-running the model.

Output columns:
    product_sku, warehouse_code, origin_date, horizon, y_true, y_pred, residual
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from core.logging import get_logger
from ml.demand.predict import DemandPredictor

log = get_logger(__name__)


def extract_residuals(
    predictor: DemandPredictor,
    feature_df: pd.DataFrame,
    *,
    max_series: Optional[int] = None,
    stride: int = 1,
) -> pd.DataFrame:
    """Produce a residual table from every valid origin in the series.

    Uses each series' entire history as origins (subject to `stride`), so
    the returned table is a superset of any split. Downstream consumers may
    filter to a date range.
    """
    rows = []
    keys = feature_df[["product_sku", "warehouse_code"]].drop_duplicates().values.tolist()
    if max_series is not None:
        keys = keys[:max_series]

    for sku, wh in keys:
        series = feature_df[
            (feature_df["product_sku"] == sku) & (feature_df["warehouse_code"] == wh)
        ].sort_values("date").reset_index(drop=True)

        T = len(series)
        for origin_idx in range(predictor.lookback - 1, T - predictor.horizon, stride):
            origin_date = pd.Timestamp(series.loc[origin_idx, "date"])
            try:
                fc = predictor.predict(series, product_sku=sku, warehouse_code=wh, origin_date=origin_date)
            except Exception:
                continue
            y_true = series.iloc[origin_idx + 1 : origin_idx + 1 + predictor.horizon]["quantity"].to_numpy(dtype=float)
            y_pred = np.array(fc.point)
            if len(y_true) != len(y_pred):
                continue
            for h in range(predictor.horizon):
                rows.append({
                    "product_sku": sku,
                    "warehouse_code": wh,
                    "origin_date": origin_date.date().isoformat(),
                    "horizon": h + 1,
                    "y_true": float(y_true[h]),
                    "y_pred": float(y_pred[h]),
                    "residual": float(y_pred[h] - y_true[h]),
                })

    df = pd.DataFrame(rows)
    log.info("demand.residuals.extracted", extra={"rows": len(df)})
    return df


def save_residuals(df: pd.DataFrame, output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p