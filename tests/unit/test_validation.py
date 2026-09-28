"""Tests for validation and cleaning."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data_platform.cleaning.demand_cleaner import clean_demand
from data_platform.cleaning.inventory_cleaner import clean_inventory
from data_platform.cleaning.supplier_cleaner import clean_supplier_events
from data_platform.validation.duplicate_detector import detect_duplicates
from data_platform.validation.missingness_report import missingness
from data_platform.validation.range_validator import RangeRule, validate_ranges
from data_platform.validation.schema_validator import SCHEMAS, validate_schema


# --- Schema ---------------
def test_schema_reports_missing_columns():
    df = pd.DataFrame({"sku": ["a"], "name": ["n"]})
    issues = validate_schema(df, SCHEMAS["products"])
    kinds = {(i.kind, i.column) for i in issues}
    assert ("missing_column", "unit_cost") in kinds


def test_schema_reports_dtype_mismatch():
    df = pd.DataFrame({
        "sku": ["a"], "name": ["n"], "description": [""], "category_code": ["X"],
        "unit_cost": ["not-a-number"], "unit_price": [1.0],
        "weight_kg": [1.0], "volume_m3": [0.01],
        "lifecycle_stage": ["maturity"], "seasonality_class": ["none"],
        "volatility_class": ["low"], "base_demand": [1.0], "lead_time_days": [7],
    })
    issues = validate_schema(df, SCHEMAS["products"])
    assert any(i.kind == "dtype_mismatch" and i.column == "unit_cost" for i in issues)


# --- Ranges ---------------
def test_range_validation_flags_negative_quantity():
    df = pd.DataFrame({"quantity": [1.0, -2.0, 3.0]})
    issues = validate_ranges(df, "demand_records", [RangeRule("quantity", minimum=0.0)])
    assert len(issues) == 1
    assert issues[0].kind == "below_min"
    assert issues[0].offending_count == 1


def test_range_validation_allowed_values():
    df = pd.DataFrame({"movement_type": ["issue", "receipt", "unknown"]})
    from data_platform.validation.range_validator import RULES
    issues = validate_ranges(df, "inventory_movements", RULES["inventory_movements"])
    assert any(i.kind == "not_allowed" for i in issues)


# --- Duplicates -----------
def test_duplicate_detector_finds_duplicate_keys():
    df = pd.DataFrame({
        "product_sku": ["A", "A", "B"],
        "warehouse_code": ["W1", "W1", "W1"],
        "demand_date": [date(2024, 1, 1)] * 3,
    })
    rep = detect_duplicates(df, "demand_records", ["product_sku", "warehouse_code", "demand_date"])
    assert rep.duplicate_rows == 2
    assert rep.distinct_duplicate_keys == 1


# --- Missingness ----------
def test_missingness_report_counts():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    r = missingness(df, "x")
    assert r.rows == 3
    assert r.columns["a"]["missing"] == 1
    assert r.columns["b"]["missing"] == 0


# --- Demand cleaner -------
def test_demand_cleaner_drops_negative_and_null_keys():
    df = pd.DataFrame({
        "product_sku": ["A", None, "B", "C"],
        "warehouse_code": ["W1", "W1", "W1", "W1"],
        "demand_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "quantity": [1.0, 2.0, -1.0, 3.0],
    })
    cleaned, summary = clean_demand(df)
    assert summary.dropped_null_keys == 1
    assert summary.dropped_negative_qty == 1
    assert len(cleaned) == 2
    assert (cleaned["quantity"] >= 0).all()


def test_demand_cleaner_flags_extreme_outliers():
    n = 100
    df = pd.DataFrame({
        "product_sku": ["A"] * (n + 1),
        "warehouse_code": ["W1"] * (n + 1),
        "demand_date": pd.date_range("2024-01-01", periods=n + 1, freq="D"),
        "quantity": list(np.random.default_rng(0).normal(10, 1, size=n)) + [10_000.0],
    })
    cleaned, summary = clean_demand(df, outlier_mad_k=5.0)
    assert summary.flagged_outliers >= 1
    assert cleaned["is_outlier"].sum() >= 1


# --- Supplier cleaner ------
def test_supplier_cleaner_drops_unknown_supplier():
    df = pd.DataFrame({
        "supplier_code": ["S1", "S2", "UNKNOWN"],
        "product_sku": ["A", "A", "A"],
        "event_type": ["delay"] * 3,
        "severity": ["low"] * 3,
        "payload": [None] * 3,
        "description": [""] * 3,
        "occurred_at": pd.to_datetime(["2024-01-01"] * 3),
    })
    cleaned, summary = clean_supplier_events(df, known_supplier_codes=["S1", "S2"])
    assert summary.dropped_unknown_supplier == 1
    assert set(cleaned["supplier_code"]) == {"S1", "S2"}


# --- Inventory cleaner -----
def test_inventory_cleaner_resolves_over_reserved():
    snaps = pd.DataFrame({
        "product_sku": ["A"], "warehouse_code": ["W1"],
        "on_hand": [10.0], "reserved": [100.0], "inbound": [0.0],
    })
    moves = pd.DataFrame()
    snaps_c, moves_c, summary = clean_inventory(snaps, moves)
    assert snaps_c.loc[0, "reserved"] == 10.0


def test_inventory_cleaner_drops_negative_snapshot():
    snaps = pd.DataFrame({
        "product_sku": ["A", "B"], "warehouse_code": ["W1", "W1"],
        "on_hand": [10.0, -5.0], "reserved": [0.0, 0.0], "inbound": [0.0, 0.0],
    })
    moves = pd.DataFrame()
    snaps_c, _, summary = clean_inventory(snaps, moves)
    assert summary.dropped_negative_snapshot == 1
    assert len(snaps_c) == 1