"""Tests for chronological split and leakage detection."""

from __future__ import annotations

import pandas as pd
import pytest

from data_platform.splits.leakage_detector import assert_no_leakage
from data_platform.splits.time_series_split import chronological_split, compute_split_dates


def _make_df(days: int = 365) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "product_sku": ["A"] * days,
        "warehouse_code": ["W1"] * days,
        "demand_date": dates,
        "quantity": range(days),
    })


def test_split_boundaries_are_ordered():
    df = _make_df()
    sd = compute_split_dates(df)
    assert sd.train_start < sd.val_start < sd.test_start <= sd.test_end


def test_chronological_split_sizes_and_boundaries():
    df = _make_df(365)
    train, val, test, sd = chronological_split(df, val_fraction=0.15, test_fraction=0.15)
    assert len(train) + len(val) + len(test) == 365
    assert train["demand_date"].max() < pd.Timestamp(sd.val_start)
    assert val["demand_date"].min() >= pd.Timestamp(sd.val_start)
    assert val["demand_date"].max() < pd.Timestamp(sd.test_start)
    assert test["demand_date"].min() >= pd.Timestamp(sd.test_start)


def test_leakage_detector_passes_on_valid_split():
    df = _make_df(365)
    train, val, test, sd = chronological_split(df)
    assert_no_leakage(train, val, test, sd)  # must not raise


def test_leakage_detector_fails_on_deliberate_leak():
    df = _make_df(365)
    train, val, test, sd = chronological_split(df)
    # Deliberately inject a training row dated inside the val window.
    leak = pd.DataFrame({
        "product_sku": ["A"],
        "warehouse_code": ["W1"],
        "demand_date": [pd.Timestamp(sd.val_start)],
        "quantity": [1],
    })
    train_leaky = pd.concat([train, leak], ignore_index=True)
    with pytest.raises(AssertionError):
        assert_no_leakage(train_leaky, val, test, sd)


def test_compute_split_dates_rejects_bad_fractions():
    df = _make_df(100)
    with pytest.raises(ValueError):
        compute_split_dates(df, val_fraction=0.6, test_fraction=0.6)
    with pytest.raises(ValueError):
        compute_split_dates(df, val_fraction=0.0, test_fraction=0.1)