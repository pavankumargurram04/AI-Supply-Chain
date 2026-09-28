"""Leakage detection for time-series splits.

Raises if:
- val_start <= train_start
- test_start <= val_start
- any training row is dated on or after `val_start`
- any validation row is dated on or after `test_start`
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from data_platform.splits.time_series_split import SplitDates


def assert_no_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    split_dates: SplitDates,
    *,
    date_col: str = "demand_date",
) -> None:
    if not (split_dates.train_start < split_dates.val_start < split_dates.test_start <= split_dates.test_end):
        raise AssertionError(
            f"Split boundary ordering violated: {split_dates.to_dict()}"
        )

    train_max = pd.to_datetime(train[date_col]).max() if len(train) else None
    val_max = pd.to_datetime(val[date_col]).max() if len(val) else None
    val_min = pd.to_datetime(val[date_col]).min() if len(val) else None
    test_min = pd.to_datetime(test[date_col]).min() if len(test) else None

    if train_max is not None and train_max >= pd.Timestamp(split_dates.val_start):
        raise AssertionError(f"Train contains rows on/after val_start: max={train_max}")
    if val_min is not None and val_min < pd.Timestamp(split_dates.val_start):
        raise AssertionError(f"Val contains rows before val_start: min={val_min}")
    if val_max is not None and val_max >= pd.Timestamp(split_dates.test_start):
        raise AssertionError(f"Val contains rows on/after test_start: max={val_max}")
    if test_min is not None and test_min < pd.Timestamp(split_dates.test_start):
        raise AssertionError(f"Test contains rows before test_start: min={test_min}")