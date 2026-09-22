"""Repository for SupplierProduct links."""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select

from db.models.supplier_product import SupplierProduct
from db.repositories.base_repository import BaseRepository


class SupplierProductRepository(BaseRepository[SupplierProduct]):
    model = SupplierProduct

    def get_pair(
        self, supplier_id: int, product_id: int
    ) -> Optional[SupplierProduct]:
        return self.get_by(supplier_id=supplier_id, product_id=product_id)

    def list_by_product(
        self, product_id: int, *, active_only: bool = True
    ) -> Sequence[SupplierProduct]:
        stmt = select(SupplierProduct).where(SupplierProduct.product_id == product_id)
        if active_only:
            stmt = stmt.where(SupplierProduct.is_active.is_(True))
        stmt = stmt.order_by(SupplierProduct.unit_price.asc())
        return list(self.session.execute(stmt).scalars().all())

    def list_by_supplier(
        self, supplier_id: int, *, active_only: bool = True
    ) -> Sequence[SupplierProduct]:
        stmt = select(SupplierProduct).where(SupplierProduct.supplier_id == supplier_id)
        if active_only:
            stmt = stmt.where(SupplierProduct.is_active.is_(True))
        stmt = stmt.order_by(SupplierProduct.product_id.asc())
        return list(self.session.execute(stmt).scalars().all())