"""Repository for PurchaseOrder and PurchaseOrderItem."""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select

from db.models.order import PurchaseOrder, PurchaseOrderItem
from db.repositories.base_repository import BaseRepository


class PurchaseOrderRepository(BaseRepository[PurchaseOrder]):
    model = PurchaseOrder

    def get_by_number(self, po_number: str) -> Optional[PurchaseOrder]:
        return self.get_by(po_number=po_number)

    def list_by_status(
        self, status: str, *, limit: int = 100, offset: int = 0
    ) -> Sequence[PurchaseOrder]:
        return self.list(
            filters={"status": status},
            order_by=PurchaseOrder.created_at.desc(),
            limit=limit,
            offset=offset,
        )

    def list_by_supplier(
        self, supplier_id: int, *, limit: int = 100, offset: int = 0
    ) -> Sequence[PurchaseOrder]:
        return self.list(
            filters={"supplier_id": supplier_id},
            order_by=PurchaseOrder.created_at.desc(),
            limit=limit,
            offset=offset,
        )

    def list_items(self, purchase_order_id: int) -> Sequence[PurchaseOrderItem]:
        stmt = (
            select(PurchaseOrderItem)
            .where(PurchaseOrderItem.purchase_order_id == purchase_order_id)
            .order_by(PurchaseOrderItem.id.asc())
        )
        return list(self.session.execute(stmt).scalars().all())