"""Immutable ledger of every inventory change."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from core.enums import MovementType
from db.base import Base, PKMixin

# No TimestampMixin: movements are append-only and immutable.


class InventoryMovement(Base, PKMixin):
    __tablename__ = "inventory_movements"

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    warehouse_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("warehouses.id", ondelete="CASCADE"), nullable=False
    )

    movement_type: Mapped[str] = mapped_column(String(32), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    unit_cost: Mapped[Optional[float]] = mapped_column(Numeric(14, 4), nullable=True)

    reference_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reference_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_inv_mov_product_wh_occurred", "product_id", "warehouse_id", "occurred_at"),
        Index("ix_inv_mov_type", "movement_type"),
        Index("ix_inv_mov_reference", "reference_type", "reference_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<InventoryMovement p={self.product_id} w={self.warehouse_id} "
            f"type={self.movement_type} qty={self.quantity}>"
        )