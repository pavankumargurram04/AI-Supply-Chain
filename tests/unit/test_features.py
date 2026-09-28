"""Tests for the feature engineering pipeline.

Focus areas:
1. Determinism: same input → same output.
2. Leakage: demand rolling/lag features must never read y[t] to build x[t].
3. Registry version is content-addressed and stable across runs.
4. Calendar features behave as expected.
5. Supplier features produce finite values where expected.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data_platform.features.demand_features import (
    add_all_demand_features,
    add_demand_lags,
    add_demand_rolling,
)
from data_platform.features.feature_registry import (
    ALL_FEATURES,
    FEATURE_VERSION,
    REGISTRY,
    feature_columns,
)
from data_platform.features.inventory_features import add_inventory_features
from data_platform.features.temporal_features import add_calendar_features


def _make_demand(n_days: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "product_sku": ["A"] * n_days,
        "warehouse_code": ["W1"] * n_days,
        "demand_date": dates,
        "quantity": np.arange(n_days, dtype=float),
        "promotion_flag": (np.arange(n_days) % 10 == 0),
        "holiday_flag": np.zeros(n_days, dtype=bool),
        "is_censored": np.zeros(n_days, dtype=bool),
    })


# --- Registry ----------
def test_registry_has_expected_kinds():
    kinds = {f.kind for f in ALL_FEATURES}
    assert {"temporal", "demand", "supplier", "inventory"} <= kinds


def test_registry_version_is_deterministic():
    # Recomputing on the same registry must yield the same version string.
    from data_platform.features.feature_registry import _compute_version
    assert _compute_version() == FEATURE_VERSION


def test_feature_columns_returns_list():
    cols = feature_columns("temporal")
    assert "feat_dow" in cols
    assert "feat_month" in cols
    assert all(c in REGISTRY for c in cols)


# --- Calendar ----------
def test_calendar_features_basic():
    df = _make_demand(10)
    out = add_calendar_features(df)
    assert out["feat_dow"].iloc[0] == 6  # 2023-01-01 is a Sunday (weekday 6)
    assert out["feat_month"].iloc[0] == 1
    assert out["feat_is_weekend"].iloc[0] == 1
    assert out["feat_is_weekend"].iloc[2] == 0  # Tuesday


def test_calendar_features_deterministic():
    df = _make_demand(20)
    a = add_calendar_features(df)
    b = add_calendar_features(df)
    pd.testing.assert_frame_equal(a, b)


# --- Lag & rolling ----------
def test_lag1_is_previous_value():
    df = _make_demand(10)
    out = add_demand_lags(df, lags=[1, 2])
    # First row has no lag.
    assert pd.isna(out["feat_demand_lag_1"].iloc[0])
    # Second row lag-1 equals first row's value.
    assert out["feat_demand_lag_1"].iloc[1] == df["quantity"].iloc[0]
    # Third row lag-2 equals first row's value.
    assert out["feat_demand_lag_2"].iloc[2] == df["quantity"].iloc[0]


def test_rolling_mean_excludes_current_row():
    """The rolling mean at t must not use y[t] — enforced by shift(1)."""
    df = _make_demand(30)
    out = add_demand_rolling(df, windows=[7])
    # Insert a huge spike at index 20. Rolling mean at index 20 must not include it.
    df2 = df.copy()
    df2.loc[20, "quantity"] = 10_000.0
    out2 = add_demand_rolling(df2, windows=[7])
    # Feature at row 20 must be identical to the version without the spike.
    v1 = out["feat_demand_roll_mean_7"].iloc[20]
    v2 = out2["feat_demand_roll_mean_7"].iloc[20]
    assert (pd.isna(v1) and pd.isna(v2)) or v1 == v2, "Leak: rolling mean at t used y[t]"

    # Feature at row 21 (the day after) SHOULD differ, since y[20] is now in its past.
    a = out["feat_demand_roll_mean_7"].iloc[21]
    b = out2["feat_demand_roll_mean_7"].iloc[21]
    assert a != b


def test_full_demand_pipeline_is_deterministic():
    df = _make_demand(120)
    a = add_all_demand_features(df)
    b = add_all_demand_features(df)
    pd.testing.assert_frame_equal(a, b)


def test_full_pipeline_no_leakage_on_spike():
    """End-to-end: inject a spike deep in the series, verify features at t are unchanged."""
    df = _make_demand(90)
    base = add_all_demand_features(df)

    spiked = df.copy()
    spiked.loc[60, "quantity"] = 1e6
    spikes = add_all_demand_features(spiked)

    # Every demand-derived feature at row 60 must be identical to the no-spike version.
    for col in base.columns:
        if not col.startswith("feat_demand_"):
            continue
        a = base[col].iloc[60]
        b = spikes[col].iloc[60]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, f"Leak detected in {col} at row 60"


# --- Inventory ----------
def test_inventory_features_derivations():
    inv = pd.DataFrame({
        "product_sku": ["A", "B"],
        "warehouse_code": ["W1", "W1"],
        "on_hand": [100.0, 50.0],
        "reserved": [20.0, 60.0],
        "inbound": [30.0, 0.0],
    })
    out = add_inventory_features(inv)
    assert out["feat_inv_available"].iloc[0] == 80.0
    assert out["feat_inv_position"].iloc[0] == 110.0
    assert out["feat_inv_reserved_ratio"].iloc[0] == pytest.approx(0.2)


# --- Supplier (via store-level wiring, minimal check here) ----------
def test_supplier_static_feature_names_present():
    static = [f.name for f in ALL_FEATURES if f.kind == "supplier"
              and f.dtype == "category"]
    assert "feat_sup_reliability" in static
    assert "feat_sup_lead_time_class" in static