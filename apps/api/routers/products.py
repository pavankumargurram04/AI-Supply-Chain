"""Product catalogue endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db
from apps.api.schemas import ProductOut
from core.errors import NotFoundError
from db.repositories.product_repository import ProductRepository

router = APIRouter(prefix="/products", tags=["products"],
                   dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[ProductOut])
def list_products(
    search: str | None = Query(default=None, description="Case-insensitive substring match"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[ProductOut]:
    repo = ProductRepository(db)
    if search:
        rows = repo.search(search, limit=limit)
    else:
        rows = repo.list_active(limit=limit, offset=offset)
    return [ProductOut.model_validate(r) for r in rows]


@router.get("/{product_id}", response_model=ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db)) -> ProductOut:
    repo = ProductRepository(db)
    row = repo.get(product_id)
    if row is None:
        raise NotFoundError(f"Product id={product_id} not found")
    return ProductOut.model_validate(row)