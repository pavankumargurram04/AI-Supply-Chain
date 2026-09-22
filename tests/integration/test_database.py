"""End-to-end verification of the schema + Alembic migration.

Asserts the full table list, foreign keys resolve, and ORM mappers configure.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import configure_mappers

import db.models  # noqa: F401  registers every model
from db.base import Base


pytestmark = pytest.mark.integration


EXPECTED_TABLES = {
    "alembic_version",
    "alerts",
    "audit_logs",
    "categories",
    "demand_records",
    "inventory",
    "inventory_movements",
    "model_registry",
    "prediction_records",
    "products",
    "purchase_order_items",
    "purchase_orders",
    "quality_events",
    "recommendation_items",
    "recommendation_runs",
    "returns",
    "scenario_runs",
    "shipment_events",
    "shipments",
    "stockouts",
    "supplier_performance_events",
    "supplier_products",
    "suppliers",
    "training_jobs",
    "users",
    "warehouses",
}


def test_all_expected_tables_exist(test_engine):
    inspector = inspect(test_engine)
    tables = set(inspector.get_table_names())
    missing = EXPECTED_TABLES - tables
    assert not missing, f"Missing tables after migration: {sorted(missing)}"


def test_mappers_configure_cleanly():
    # This raises if any relationship or column annotation is malformed.
    configure_mappers()
    assert len(Base.metadata.tables) == len(EXPECTED_TABLES) - 1  # exclude alembic_version


def test_foreign_keys_present_on_inventory(test_engine):
    inspector = inspect(test_engine)
    fks = inspector.get_foreign_keys("inventory")
    cols = {fk["constrained_columns"][0] for fk in fks}
    assert {"product_id", "warehouse_id"} <= cols


def test_unique_inventory_pair_exists(test_engine):
    inspector = inspect(test_engine)
    uniques = inspector.get_unique_constraints("inventory")
    columns = {tuple(sorted(u["column_names"])) for u in uniques}
    assert ("product_id", "warehouse_id") in columns


def test_demand_unique_series_date_exists(test_engine):
    inspector = inspect(test_engine)
    uniques = inspector.get_unique_constraints("demand_records")
    columns = {tuple(sorted(u["column_names"])) for u in uniques}
    assert ("demand_date", "product_id", "warehouse_id") in columns