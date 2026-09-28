"""Chronological train / validation / test split.

For every (product, warehouse) series, all three splits must be
contiguous in time and non-overlapping:

    train: [start, cut_val)
    val:   [cut_val, cut_test)
    test:  [cut_test, end]

Where `cut_val` and `cut_test` are computed as global calendar cut dates
so all series share the same boundary (required for a global model).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

import pandas as pd


@dataclass
class SplitDates:
    train_start: date
    val_start: date
    test_start: date
    test_end: date

    def to_dict(self) -> dict:
        return {
            "train_start": self.train_start.isoformat(),
            "val_start": self.val_start.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def _dates(dates: pd.Series) -> Tuple[date, date]:
    return (
        pd.to_datetime(dates.min()).date(),
        pd.to_datetime(dates.max()).date(),
    )


def compute_split_dates(
    df: pd.DataFrame,
    *,
    date_col: str = "demand_date",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitDates:
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("val_fraction and test_fraction must be positive.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.")

    dmin, dmax = _dates(df[date_col])
    total_days = (dmax - dmin).days + 1

    n_val = max(1, int(round(total_days * val_fraction)))
    n_test = max(1, int(round(total_days * test_fraction)))
    n_train = total_days - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough days to construct a non-empty train split.")

    val_start = dmin + pd.Timedelta(days=n_train).to_pytimedelta()  # type: ignore[attr-defined]
    test_start = val_start + pd.Timedelta(days=n_val).to_pytimedelta()  # type: ignore[attr-defined]

    return SplitDates(
        train_start=dmin,
        val_start=val_start,
        test_start=test_start,
        test_end=dmax,
    )


def chronological_split(
    df: pd.DataFrame,
    *,
    date_col: str = "demand_date",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    split_dates: Optional[SplitDates] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitDates]:
    """Return (train, val, test, SplitDates)."""
    if split_dates is None:
        split_dates = compute_split_dates(
            df, date_col=date_col,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

    d = pd.to_datetime(df[date_col])
    train = df[d < pd.Timestamp(split_dates.val_start)].copy()
    val = df[
        (d >= pd.Timestamp(split_dates.val_start)) & (d < pd.Timestamp(split_dates.test_start))
    ].copy()
    test = df[d >= pd.Timestamp(split_dates.test_start)].copy()

    return train, val, test, split_dates