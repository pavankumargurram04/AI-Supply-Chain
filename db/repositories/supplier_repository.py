"""Repository for Supplier."""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select

from db.models.supplier import Supplier
from db.repositories.base_repository import BaseRepository


class SupplierRepository(BaseRepository[Supplier]):
    model = Supplier

    def get_by_code(self, code: str) -> Optional[Supplier]:
        return self.get_by(code=code)

    def list_active(
        self, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Supplier]:
        return self.list(
            filters={"is_active": True},
            order_by=Supplier.name.asc(),
            limit=limit,
            offset=offset,
        )

    def list_by_reliability(self, reliability_class: str) -> Sequence[Supplier]:
        stmt = (
            select(Supplier)
            .where(
                Supplier.reliability_class == reliability_class,
                Supplier.is_active.is_(True),
            )
            .order_by(Supplier.name.asc())
        )
        return list(self.session.execute(stmt).scalars().all())