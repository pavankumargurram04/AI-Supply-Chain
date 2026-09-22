"""Customer / downstream returns."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, PKMixin


class Return(Base, PKMixin):
    __tablename__ = "returns"

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    warehouse_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("warehouses.id", ondelete="CASCADE"), nullable=False
    )
    supplier_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("suppliers.id", ondelete="SET NULL"), nullable=True
    )

    quantity: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    unit_cost: Mapped[Optional[float]] = mapped_column(Numeric(14, 4), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="received")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_returns_product_occurred", "product_id", "occurred_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Return p={self.product_id} w={self.warehouse_id} qty={self.quantity}>"