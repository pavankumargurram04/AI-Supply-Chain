"""Repository for Product."""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import or_, select

from db.models.product import Product
from db.repositories.base_repository import BaseRepository


class ProductRepository(BaseRepository[Product]):
    model = Product

    def get_by_sku(self, sku: str) -> Optional[Product]:
        return self.get_by(sku=sku)

    def list_active(
        self, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Product]:
        return self.list(
            filters={"is_active": True},
            order_by=Product.sku.asc(),
            limit=limit,
            offset=offset,
        )

    def search(
        self, query: str, *, limit: int = 50
    ) -> Sequence[Product]:
        """Case-insensitive substring search on SKU or name."""
        like = f"%{query.lower()}%"
        stmt = (
            select(Product)
            .where(
                or_(
                    Product.sku.ilike(like),
                    Product.name.ilike(like),
                )
            )
            .order_by(Product.sku.asc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def list_by_category(
        self, category_id: int, *, limit: int = 200, offset: int = 0
    ) -> Sequence[Product]:
        return self.list(
            filters={"category_id": category_id, "is_active": True},
            order_by=Product.sku.asc(),
            limit=limit,
            offset=offset,
        )