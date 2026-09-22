"""Product categories (self-referencing hierarchy)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import ActiveMixin, Base, PKMixin, TimestampMixin

if TYPE_CHECKING:
    from db.models.product import Product


class Category(Base, PKMixin, TimestampMixin, ActiveMixin):
    __tablename__ = "categories"

    code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parent_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("categories.id", ondelete="SET NULL"), nullable=True
    )

    parent: Mapped[Optional["Category"]] = relationship(
        "Category", remote_side="Category.id", backref="children"
    )
    products: Mapped[list["Product"]] = relationship(
        "Product", back_populates="category", cascade="save-update"
    )

    __table_args__ = (
        Index("ix_categories_parent_id", "parent_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Category id={self.id} code={self.code!r}>"