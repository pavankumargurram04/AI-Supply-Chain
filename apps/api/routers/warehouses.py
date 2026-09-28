"""Warehouse endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db
from apps.api.schemas import WarehouseOut
from core.errors import NotFoundError
from db.repositories.warehouse_repository import WarehouseRepository

router = APIRouter(prefix="/warehouses", tags=["warehouses"],
                   dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[WarehouseOut])
def list_warehouses(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[WarehouseOut]:
    repo = WarehouseRepository(db)
    rows = repo.list_active(limit=limit, offset=offset)
    return [WarehouseOut.model_validate(r) for r in rows]


@router.get("/{warehouse_id}", response_model=WarehouseOut)
def get_warehouse(warehouse_id: int, db: Session = Depends(get_db)) -> WarehouseOut:
    repo = WarehouseRepository(db)
    row = repo.get(warehouse_id)
    if row is None:
        raise NotFoundError(f"Warehouse id={warehouse_id} not found")
    return WarehouseOut.model_validate(row)