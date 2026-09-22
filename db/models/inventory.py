"""Inventory snapshot (one row per product/warehouse pair)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Numeric, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, PKMixin, TimestampMixin

if TYPE_CHECKING:
    from db.models.product import Product
    from db.models.warehouse import Warehouse


class Inventory(Base, PKMixin, TimestampMixin):
    __tablename__ = "inventory"

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    warehouse_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("warehouses.id", ondelete="CASCADE"), nullable=False
    )

    on_hand: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False, default=0.0)
    reserved: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False, default=0.0)
    inbound: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False, default=0.0)

    # Cached latest computed values (recomputed by decision engine).
    safety_stock: Mapped[Optional[float]] = mapped_column(Numeric(16, 4), nullable=True)
    reorder_point: Mapped[Optional[float]] = mapped_column(Numeric(16, 4), nullable=True)

    last_counted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    product: Mapped["Product"] = relationship("Product", back_populates="inventories")
    warehouse: Mapped["Warehouse"] = relationship("Warehouse", back_populates="inventories")

    @property
    def available(self) -> float:
        return float(self.on_hand) - float(self.reserved)

    @property
    def inventory_position(self) -> float:
        return float(self.on_hand) - float(self.reserved) + float(self.inbound)

    __table_args__ = (
        UniqueConstraint("product_id", "warehouse_id", name="uq_inventory_pair"),
        Index("ix_inventory_warehouse_id", "warehouse_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Inventory p={self.product_id} w={self.warehouse_id} on_hand={self.on_hand}>"