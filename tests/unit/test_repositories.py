"""Repository smoke tests using the real Postgres test DB.

If Postgres is unavailable the whole module is skipped.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.enums import UserRole
from core.security import hash_password
from db.models.category import Category
from db.models.demand import DemandRecord
from db.models.inventory import Inventory
from db.models.product import Product
from db.models.warehouse import Warehouse
from db.repositories.category_repository import CategoryRepository
from db.repositories.demand_repository import DemandRepository
from db.repositories.inventory_repository import InventoryRepository
from db.repositories.product_repository import ProductRepository
from db.repositories.user_repository import UserRepository
from db.repositories.warehouse_repository import WarehouseRepository


pytestmark = pytest.mark.integration


def test_category_create_and_get(db_session):
    repo = CategoryRepository(db_session)
    cat = repo.add(Category(code="ELEC", name="Electronics"))
    db_session.flush()
    assert repo.get_by_code("ELEC").id == cat.id
    assert repo.exists(code="ELEC") is True


def test_product_creation_with_category(db_session):
    cats = CategoryRepository(db_session)
    prods = ProductRepository(db_session)
    c = cats.add(Category(code="TOOLS", name="Tools"))
    db_session.flush()
    p = prods.add(
        Product(
            sku="SKU-1",
            name="Hammer",
            category_id=c.id,
            unit_cost=5.0,
            unit_price=12.0,
            base_demand=10.0,
            lead_time_days=7,
        )
    )
    db_session.flush()
    got = prods.get_by_sku("SKU-1")
    assert got is not None
    assert got.category_id == c.id


def test_product_search_is_case_insensitive(db_session):
    prods = ProductRepository(db_session)
    prods.add(
        Product(
            sku="SKU-ABC",
            name="Blue Widget",
            unit_cost=1.0,
            unit_price=3.0,
            base_demand=1.0,
            lead_time_days=3,
        )
    )
    db_session.flush()
    hits_lower = prods.search("widget")
    hits_upper = prods.search("WIDGET")
    assert {p.id for p in hits_lower} == {p.id for p in hits_upper}
    assert len(hits_lower) >= 1


def test_inventory_available_and_position(db_session):
    prods = ProductRepository(db_session)
    whs = WarehouseRepository(db_session)
    inv = InventoryRepository(db_session)

    p = prods.add(
        Product(
            sku="SKU-INV",
            name="Inventory Target",
            unit_cost=10.0,
            unit_price=25.0,
            base_demand=5.0,
            lead_time_days=5,
        )
    )
    w = whs.add(Warehouse(code="WH-A", name="Warehouse A", capacity_units=100000.0))
    db_session.flush()

    row = inv.add(
        Inventory(product_id=p.id, warehouse_id=w.id, on_hand=100, reserved=20, inbound=30)
    )
    db_session.flush()

    assert row.available == 80
    assert row.inventory_position == 110


def test_demand_bulk_upsert(db_session):
    prods = ProductRepository(db_session)
    whs = WarehouseRepository(db_session)
    dem = DemandRepository(db_session)

    p = prods.add(
        Product(
            sku="SKU-D1",
            name="Demand",
            unit_cost=1.0,
            unit_price=2.0,
            base_demand=1.0,
            lead_time_days=3,
        )
    )
    w = whs.add(Warehouse(code="WH-D1", name="Demand WH", capacity_units=10_000.0))
    db_session.flush()

    today = date(2024, 1, 1)
    rows = [
        {
            "product_id": p.id,
            "warehouse_id": w.id,
            "demand_date": today + timedelta(days=i),
            "quantity": 10.0 + i,
            "unit_price": 2.0,
            "promotion_flag": False,
            "holiday_flag": False,
            "is_censored": False,
            "source": "synthetic",
        }
        for i in range(5)
    ]
    inserted = dem.bulk_upsert(rows)
    assert inserted == 5
    total = dem.total_for_series(p.id, w.id)
    assert total == sum(10 + i for i in range(5))

    # Idempotent: upsert same rows, values are updated (not duplicated).
    rows[0]["quantity"] = 999.0
    dem.bulk_upsert(rows)
    assert dem.total_for_series(p.id, w.id) == 999.0 + sum(10 + i for i in range(1, 5))
    assert dem.count(filters={"product_id": p.id}) == 5


def test_user_repository_hashes_are_stored(db_session):
    repo = UserRepository(db_session)
    user = repo.create(
        email="a@b.com",
        username="tester",
        hashed_password=hash_password("pw"),
        role=UserRole.ANALYST.value,
    )
    db_session.flush()
    assert repo.get_by_email("A@B.COM").id == user.id
    assert repo.get_by_username("tester").id == user.id