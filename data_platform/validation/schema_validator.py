"""Schema validation: column presence, order, and dtype checks.

A `Schema` describes the expected shape of one logical table. Every
dataset produced by the generator or ingested from external sources must
pass the schema before downstream processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Schema:
    name: str
    required_columns: Dict[str, str] = field(default_factory=dict)  # col -> expected dtype family
    allow_extra_columns: bool = True


@dataclass
class SchemaIssue:
    table: str
    kind: str          # "missing_column" | "extra_column" | "dtype_mismatch"
    column: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"table": self.table, "kind": self.kind, "column": self.column, "detail": self.detail}


_DTYPE_FAMILY = {
    "int": {"int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"},
    "float": {"float16", "float32", "float64"},
    "bool": {"bool", "boolean"},
    "datetime": {"datetime64[ns]", "datetime64[ns, UTC]"},
    "object": {"object", "string"},
    "category": {"category"},
}


def _family_of(dtype) -> str:
    s = str(dtype)
    for family, values in _DTYPE_FAMILY.items():
        if s in values or s.startswith(tuple(values)):
            return family
    return "unknown"


def validate_schema(df: pd.DataFrame, schema: Schema) -> List[SchemaIssue]:
    issues: List[SchemaIssue] = []

    for col, family in schema.required_columns.items():
        if col not in df.columns:
            issues.append(SchemaIssue(schema.name, "missing_column", col, f"expected family={family}"))
            continue
        actual = _family_of(df[col].dtype)
        if actual != family:
            # Allow int columns to satisfy float requirements (numeric widening).
            if family == "float" and actual == "int":
                continue
            issues.append(
                SchemaIssue(
                    schema.name,
                    "dtype_mismatch",
                    col,
                    f"expected={family}, got={actual} ({df[col].dtype})",
                )
            )

    if not schema.allow_extra_columns:
        for col in df.columns:
            if col not in schema.required_columns:
                issues.append(SchemaIssue(schema.name, "extra_column", col, ""))

    return issues


# ---------------------------------------------------------------------------
# Canonical schemas for the generated dataset tables.
# ---------------------------------------------------------------------------

SCHEMAS: Dict[str, Schema] = {
    "categories": Schema(
        name="categories",
        required_columns={
            "code": "object",
            "name": "object",
            "description": "object",
            "parent_code": "object",
        },
    ),
    "products": Schema(
        name="products",
        required_columns={
            "sku": "object",
            "name": "object",
            "description": "object",
            "category_code": "object",
            "unit_cost": "float",
            "unit_price": "float",
            "weight_kg": "float",
            "volume_m3": "float",
            "lifecycle_stage": "object",
            "seasonality_class": "object",
            "volatility_class": "object",
            "base_demand": "float",
            "lead_time_days": "int",
        },
    ),
    "suppliers": Schema(
        name="suppliers",
        required_columns={
            "code": "object",
            "name": "object",
            "contact_email": "object",
            "contact_phone": "object",
            "country": "object",
            "region": "object",
            "reliability_class": "object",
            "cost_class": "object",
            "quality_class": "object",
            "lead_time_class": "object",
            "capacity_class": "object",
            "contracted_capacity_units": "float",
            "notes": "object",
        },
    ),
    "warehouses": Schema(
        name="warehouses",
        required_columns={
            "code": "object",
            "name": "object",
            "region": "object",
            "country": "object",
            "latitude": "float",
            "longitude": "float",
            "capacity_units": "float",
            "operating_cost_daily": "float",
        },
    ),
    "supplier_products": Schema(
        name="supplier_products",
        required_columns={
            "supplier_code": "object",
            "product_sku": "object",
            "unit_price": "float",
            "moq": "int",
            "max_order_qty": "int",
            "lead_time_days": "float",
            "lead_time_std_days": "float",
            "shipping_cost": "float",
            "is_preferred": "bool",
        },
    ),
    "supplier_events": Schema(
        name="supplier_events",
        required_columns={
            "supplier_code": "object",
            "product_sku": "object",
            "event_type": "object",
            "severity": "object",
            "payload": "object",
            "description": "object",
            "occurred_at": "datetime",
        },
    ),
    "demand_records": Schema(
        name="demand_records",
        required_columns={
            "product_sku": "object",
            "warehouse_code": "object",
            "demand_date": "datetime",
            "quantity": "float",
            "promotion_flag": "bool",
            "holiday_flag": "bool",
            "is_censored": "bool",
            "source": "object",
        },
        # yearly/monthly partition columns may appear — allow extras.
    ),
    "inventory": Schema(
        name="inventory",
        required_columns={
            "product_sku": "object",
            "warehouse_code": "object",
            "on_hand": "float",
            "reserved": "float",
            "inbound": "float",
        },
    ),
    "inventory_movements": Schema(
        name="inventory_movements",
        required_columns={
            "product_sku": "object",
            "warehouse_code": "object",
            "movement_type": "object",
            "quantity": "float",
            "occurred_at": "datetime",
        },
    ),
}