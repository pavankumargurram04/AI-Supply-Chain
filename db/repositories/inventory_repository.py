"""Repository for Inventory snapshots."""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select

from db.models.inventory import Inventory
from db.repositories.base_repository import BaseRepository


class InventoryRepository(BaseRepository[Inventory]):
    model = Inventory

    def get_for_pair(self, product_id: int, warehouse_id: int) -> Optional[Inventory]:
        return self.get_by(product_id=product_id, warehouse_id=warehouse_id)

    def list_for_product(self, product_id: int) -> Sequence[Inventory]:
        stmt = (
            select(Inventory)
            .where(Inventory.product_id == product_id)
            .order_by(Inventory.warehouse_id.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def list_for_warehouse(
        self, warehouse_id: int, *, limit: int = 500, offset: int = 0
    ) -> Sequence[Inventory]:
        return self.list(
            filters={"warehouse_id": warehouse_id},
            order_by=Inventory.product_id.asc(),
            limit=limit,
            offset=offset,
        )

    def list_low_stock(self, *, threshold: float = 0.0) -> Sequence[Inventory]:
        """Rows where on_hand is at or below a raw threshold.

        NOTE: This helper is intentionally a raw *listing* convenience. It is
        NOT a decision rule. Real stockout intelligence is probabilistic and
        lives in `decision_engine/` — never in this repository.
        """
        stmt = select(Inventory).where(Inventory.on_hand <= threshold)
        return list(self.session.execute(stmt).scalars().all())