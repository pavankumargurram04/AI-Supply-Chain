"""Cleaning rules for demand records.

Cleaning steps (all conservative and logged):
1. Drop rows with null keys or null demand date.
2. Coerce dtypes: quantities to float, dates to datetime.
3. Remove negative quantities (keep zero — legitimate no-demand days).
4. Flag rows as `is_outlier` where quantity > median + k * MAD within series.
5. Return cleaned DataFrame plus a summary.

Censored demand handling (replacing censored rows with an estimate) is
deliberately NOT done here — that is a modelling decision and belongs in
the demand modelling phase, with full provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class CleanSummary:
    table: str
    rows_in: int
    rows_out: int
    dropped_null_keys: int = 0
    dropped_negative_qty: int = 0
    coerced_dates: int = 0
    flagged_outliers: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped_null_keys": self.dropped_null_keys,
            "dropped_negative_qty": self.dropped_negative_qty,
            "coerced_dates": self.coerced_dates,
            "flagged_outliers": self.flagged_outliers,
            "notes": self.notes,
        }


def clean_demand(
    df: pd.DataFrame,
    *,
    outlier_mad_k: float = 6.0,
) -> tuple[pd.DataFrame, CleanSummary]:
    summary = CleanSummary(table="demand_records", rows_in=len(df), rows_out=0)
    if df.empty:
        return df, summary

    df = df.copy()

    # 1) Null keys.
    key_cols = ["product_sku", "warehouse_code", "demand_date"]
    before = len(df)
    df = df.dropna(subset=key_cols)
    summary.dropped_null_keys = before - len(df)

    # 2) Dtypes.
    if not np.issubdtype(df["demand_date"].dtype, np.datetime64):
        df["demand_date"] = pd.to_datetime(df["demand_date"], errors="coerce")
        summary.coerced_dates = int(df["demand_date"].isna().sum())
        df = df.dropna(subset=["demand_date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["quantity"])

    # 3) Negative quantities.
    neg_mask = df["quantity"] < 0
    summary.dropped_negative_qty = int(neg_mask.sum())
    df = df[~neg_mask]

    # 4) Outlier flag (MAD-based per series).
    if len(df) > 0:
        med = df.groupby(["product_sku", "warehouse_code"])["quantity"].transform("median")
        mad = df.groupby(["product_sku", "warehouse_code"])["quantity"].transform(
            lambda x: (x - x.median()).abs().median()
        )
        # When MAD is zero (constant series), avoid division; fall back to a small epsilon.
        mad = mad.replace(0.0, np.nan)
        robust_z = (df["quantity"] - med) / (1.4826 * mad)
        df["is_outlier"] = (robust_z.abs() > outlier_mad_k).fillna(False)
        summary.flagged_outliers = int(df["is_outlier"].sum())
        summary.notes.append(f"MAD outlier threshold = {outlier_mad_k}")
    else:
        df["is_outlier"] = False

    summary.rows_out = len(df)
    return df.reset_index(drop=True), summary