"""Supplier endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db
from apps.api.schemas import SupplierOut
from core.errors import NotFoundError
from db.repositories.supplier_repository import SupplierRepository

router = APIRouter(prefix="/suppliers", tags=["suppliers"],
                   dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[SupplierOut])
def list_suppliers(
    reliability: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[SupplierOut]:
    repo = SupplierRepository(db)
    if reliability:
        rows = repo.list_by_reliability(reliability)
    else:
        rows = repo.list_active(limit=limit, offset=offset)
    return [SupplierOut.model_validate(r) for r in rows]


@router.get("/{supplier_id}", response_model=SupplierOut)
def get_supplier(supplier_id: int, db: Session = Depends(get_db)) -> SupplierOut:
    repo = SupplierRepository(db)
    row = repo.get(supplier_id)
    if row is None:
        raise NotFoundError(f"Supplier id={supplier_id} not found")
    return SupplierOut.model_validate(row)