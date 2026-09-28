"""Loads generated Parquet tables into PostgreSQL.

Resolves natural keys (SKU / code) to database IDs at ingest time.

Uses SQLAlchemy Core `insert().values()` in batches for speed with PostgreSQL
ON CONFLICT DO NOTHING for safe re-entrancy / idempotency.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from core.logging import get_logger
from db.models.category import Category
from db.models.demand import DemandRecord
from db.models.inventory import Inventory
from db.models.inventory_movement import InventoryMovement
from db.models.product import Product
from db.models.supplier import Supplier
from db.models.supplier_event import SupplierEvent
from db.models.supplier_product import SupplierProduct
from db.models.warehouse import Warehouse

log = get_logger(__name__)


def _load_lookup(session: Session, model: Any, key_col: str, id_col: str = "id") -> Dict[Any, int]:
    stmt = select(getattr(model, key_col), getattr(model, id_col))
    return {row[0]: row[1] for row in session.execute(stmt).all()}


def ingest_categories(session: Session, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [
        {"code": r.code, "name": r.name, "description": r.description}
        for r in df.itertuples(index=False)
    ]
    stmt = insert(Category).values(rows).on_conflict_do_nothing(index_elements=["code"])
    session.execute(stmt)
    session.flush()
    return len(rows)


def ingest_products(session: Session, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cat_lookup = _load_lookup(session, Category, "code")
    rows: List[Dict[str, Any]] = []
    for r in df.itertuples(index=False):
        rows.append({
            "sku": r.sku,
            "name": r.name,
            "description": r.description,
            "category_id": cat_lookup.get(r.category_code),
            "unit_cost": float(r.unit_cost),
            "unit_price": float(r.unit_price),
            "weight_kg": float(r.weight_kg),
            "volume_m3": float(r.volume_m3),
            "lifecycle_stage": r.lifecycle_stage,
            "seasonality_class": r.seasonality_class,
            "volatility_class": r.volatility_class,
            "base_demand": float(r.base_demand),
            "lead_time_days": int(r.lead_time_days),
        })
    stmt = insert(Product).values(rows).on_conflict_do_nothing(index_elements=["sku"])
    session.execute(stmt)
    session.flush()
    return len(rows)


def ingest_suppliers(session: Session, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [{
        "code": r.code,
        "name": r.name,
        "contact_email": r.contact_email,
        "contact_phone": r.contact_phone,
        "country": r.country,
        "region": r.region,
        "reliability_class": r.reliability_class,
        "cost_class": r.cost_class,
        "quality_class": r.quality_class,
        "lead_time_class": r.lead_time_class,
        "capacity_class": r.capacity_class,
        "contracted_capacity_units": float(r.contracted_capacity_units),
        "notes": r.notes,
    } for r in df.itertuples(index=False)]
    stmt = insert(Supplier).values(rows).on_conflict_do_nothing(index_elements=["code"])
    session.execute(stmt)
    session.flush()
    return len(rows)


def ingest_warehouses(session: Session, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [{
        "code": r.code,
        "name": r.name,
        "region": r.region,
        "country": r.country,
        "latitude": float(r.latitude) if r.latitude is not None else None,
        "longitude": float(r.longitude) if r.longitude is not None else None,
        "capacity_units": float(r.capacity_units),
        "operating_cost_daily": float(r.operating_cost_daily),
    } for r in df.itertuples(index=False)]
    stmt = insert(Warehouse).values(rows).on_conflict_do_nothing(index_elements=["code"])
    session.execute(stmt)
    session.flush()
    return len(rows)


def ingest_supplier_products(session: Session, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    sup_lookup = _load_lookup(session, Supplier, "code")
    prod_lookup = _load_lookup(session, Product, "sku")
    rows: List[Dict[str, Any]] = []
    for r in df.itertuples(index=False):
        sid = sup_lookup.get(r.supplier_code)
        pid = prod_lookup.get(r.product_sku)
        if sid is None or pid is None:
            continue
        rows.append({
            "supplier_id": sid,
            "product_id": pid,
            "unit_price": float(r.unit_price),
            "moq": int(r.moq),
            "max_order_qty": int(r.max_order_qty),
            "lead_time_days": float(r.lead_time_days),
            "lead_time_std_days": float(r.lead_time_std_days),
            "shipping_cost": float(r.shipping_cost),
            "is_preferred": bool(r.is_preferred),
        })
    if not rows:
        return 0
    stmt = insert(SupplierProduct).values(rows).on_conflict_do_nothing(index_elements=["supplier_id", "product_id"])
    session.execute(stmt)
    session.flush()
    return len(rows)


def ingest_demand(session: Session, df: pd.DataFrame, *, batch_size: int = 10_000) -> int:
    if df.empty:
        return 0
    prod_lookup = _load_lookup(session, Product, "sku")
    wh_lookup = _load_lookup(session, Warehouse, "code")

    total = 0
    buf: List[Dict[str, Any]] = []
    for r in df.itertuples(index=False):
        pid = prod_lookup.get(r.product_sku)
        wid = wh_lookup.get(r.warehouse_code)
        if pid is None or wid is None:
            continue
        buf.append({
            "product_id": pid,
            "warehouse_id": wid,
            "demand_date": r.demand_date,
            "quantity": float(r.quantity),
            "unit_price": None,
            "promotion_flag": bool(r.promotion_flag),
            "holiday_flag": bool(r.holiday_flag),
            "is_censored": bool(r.is_censored),
            "source": r.source,
        })
        if len(buf) >= batch_size:
            stmt = insert(DemandRecord).values(buf).on_conflict_do_nothing()
            session.execute(stmt)
            session.flush()
            total += len(buf)
            buf.clear()
    if buf:
        stmt = insert(DemandRecord).values(buf).on_conflict_do_nothing()
        session.execute(stmt)
        session.flush()
        total += len(buf)
    return total


def ingest_inventory(session: Session, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    prod_lookup = _load_lookup(session, Product, "sku")
    wh_lookup = _load_lookup(session, Warehouse, "code")
    rows: List[Dict[str, Any]] = []
    for r in df.itertuples(index=False):
        pid = prod_lookup.get(r.product_sku)
        wid = wh_lookup.get(r.warehouse_code)
        if pid is None or wid is None:
            continue
        rows.append({
            "product_id": pid,
            "warehouse_id": wid,
            "on_hand": float(r.on_hand),
            "reserved": float(r.reserved),
            "inbound": float(r.inbound),
            "safety_stock": None,
            "reorder_point": None,
            "last_counted_at": None,
        })
    if not rows:
        return 0
    stmt = insert(Inventory).values(rows).on_conflict_do_nothing(index_elements=["product_id", "warehouse_id"])
    session.execute(stmt)
    session.flush()
    return len(rows)


def ingest_inventory_movements(session: Session, df: pd.DataFrame, *, batch_size: int = 10_000) -> int:
    if df.empty:
        return 0
    prod_lookup = _load_lookup(session, Product, "sku")
    wh_lookup = _load_lookup(session, Warehouse, "code")

    total = 0
    buf: List[Dict[str, Any]] = []
    for r in df.itertuples(index=False):
        pid = prod_lookup.get(r.product_sku)
        wid = wh_lookup.get(r.warehouse_code)
        if pid is None or wid is None:
            continue
        buf.append({
            "product_id": pid,
            "warehouse_id": wid,
            "movement_type": r.movement_type,
            "quantity": float(r.quantity),
            "unit_cost": None,
            "reference_type": r.reference_type if isinstance(r.reference_type, str) else None,
            "reference_id": None,
            "notes": r.notes,
            "occurred_at": r.occurred_at,
        })
        if len(buf) >= batch_size:
            stmt = insert(InventoryMovement).values(buf).on_conflict_do_nothing()
            session.execute(stmt)
            session.flush()
            total += len(buf)
            buf.clear()
    if buf:
        stmt = insert(InventoryMovement).values(buf).on_conflict_do_nothing()
        session.execute(stmt)
        session.flush()
        total += len(buf)
    return total


def ingest_supplier_events(session: Session, df: pd.DataFrame, *, batch_size: int = 5_000) -> int:
    if df.empty:
        return 0
    sup_lookup = _load_lookup(session, Supplier, "code")
    prod_lookup = _load_lookup(session, Product, "sku")

    total = 0
    buf: List[Dict[str, Any]] = []
    for r in df.itertuples(index=False):
        sid = sup_lookup.get(r.supplier_code)
        if sid is None:
            continue
        pid = prod_lookup.get(r.product_sku) if r.product_sku else None
        buf.append({
            "supplier_id": sid,
            "product_id": pid,
            "event_type": r.event_type,
            "severity": r.severity,
            "payload": r.payload if isinstance(r.payload, dict) else None,
            "description": r.description,
            "occurred_at": r.occurred_at,
        })
        if len(buf) >= batch_size:
            stmt = insert(SupplierEvent).values(buf).on_conflict_do_nothing()
            session.execute(stmt)
            session.flush()
            total += len(buf)
            buf.clear()
    if buf:
        stmt = insert(SupplierEvent).values(buf).on_conflict_do_nothing()
        session.execute(stmt)
        session.flush()
        total += len(buf)
    return total