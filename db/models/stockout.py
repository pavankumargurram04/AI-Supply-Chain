"""Recorded stockout events (for downstream evaluation)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, PKMixin


class Stockout(Base, PKMixin):
    __tablename__ = "stockouts"

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    warehouse_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("warehouses.id", ondelete="CASCADE"), nullable=False
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_hours: Mapped[Optional[float]] = mapped_column(Numeric(10, 2), nullable=True)

    lost_demand_units: Mapped[Optional[float]] = mapped_column(Numeric(16, 4), nullable=True)
    estimated_lost_margin: Mapped[Optional[float]] = mapped_column(Numeric(16, 4), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_stockouts_product_wh_started", "product_id", "warehouse_id", "started_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Stockout p={self.product_id} w={self.warehouse_id} at={self.started_at}>"