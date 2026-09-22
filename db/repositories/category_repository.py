"""Repository for Category."""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select

from db.models.category import Category
from db.repositories.base_repository import BaseRepository


class CategoryRepository(BaseRepository[Category]):
    model = Category

    def get_by_code(self, code: str) -> Optional[Category]:
        return self.get_by(code=code)

    def list_active(self, limit: int = 500, offset: int = 0) -> Sequence[Category]:
        return self.list(
            filters={"is_active": True},
            order_by=Category.name.asc(),
            limit=limit,
            offset=offset,
        )

    def list_roots(self) -> Sequence[Category]:
        stmt = (
            select(Category)
            .where(Category.parent_id.is_(None))
            .order_by(Category.name.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def children_of(self, parent_id: int) -> Sequence[Category]:
        stmt = (
            select(Category)
            .where(Category.parent_id == parent_id)
            .order_by(Category.name.asc())
        )
        return list(self.session.execute(stmt).scalars().all())