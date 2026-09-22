"""Repository for Warehouse."""

from __future__ import annotations

from typing import Optional, Sequence

from db.models.warehouse import Warehouse
from db.repositories.base_repository import BaseRepository


class WarehouseRepository(BaseRepository[Warehouse]):
    model = Warehouse

    def get_by_code(self, code: str) -> Optional[Warehouse]:
        return self.get_by(code=code)

    def list_active(
        self, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Warehouse]:
        return self.list(
            filters={"is_active": True},
            order_by=Warehouse.name.asc(),
            limit=limit,
            offset=offset,
        )