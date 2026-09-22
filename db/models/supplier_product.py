"""Supplier–product association with commercial terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, ForeignKey, Index, Numeric, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import ActiveMixin, Base, PKMixin, TimestampMixin

if TYPE_CHECKING:
    from db.models.product import Product
    from db.models.supplier import Supplier


class SupplierProduct(Base, PKMixin, TimestampMixin, ActiveMixin):
    __tablename__ = "supplier_products"

    supplier_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("suppliers.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )

    unit_price: Mapped[float] = mapped_column(Numeric(14, 4), nullable=False)
    moq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    max_order_qty: Mapped[int] = mapped_column(BigInteger, nullable=False, default=100000)
    lead_time_days: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False, default=7.0)
    lead_time_std_days: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False, default=1.5)
    shipping_cost: Mapped[float] = mapped_column(Numeric(14, 4), nullable=False, default=0.0)
    is_preferred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    supplier: Mapped["Supplier"] = relationship("Supplier", back_populates="product_links")
    product: Mapped["Product"] = relationship("Product", back_populates="supplier_links")

    __table_args__ = (
        UniqueConstraint("supplier_id", "product_id", name="uq_supplier_products_pair"),
        Index("ix_supplier_products_product_id", "product_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SupplierProduct s={self.supplier_id} p={self.product_id}>"