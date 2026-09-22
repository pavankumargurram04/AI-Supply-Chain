"""Supplier performance events (delays, outages, price changes, ...)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, PKMixin

if TYPE_CHECKING:
    from db.models.supplier import Supplier


class SupplierEvent(Base, PKMixin):
    __tablename__ = "supplier_performance_events"

    supplier_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("suppliers.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="SET NULL"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="medium")

    # Numeric context, e.g. {"delay_days": 3, "quantity_short": 40}
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    supplier: Mapped["Supplier"] = relationship("Supplier", back_populates="performance_events")

    __table_args__ = (
        Index("ix_supplier_events_supplier_occurred", "supplier_id", "occurred_at"),
        Index("ix_supplier_events_type", "event_type"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SupplierEvent s={self.supplier_id} type={self.event_type}>"