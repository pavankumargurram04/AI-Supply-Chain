"""Quality events (inspection failures, defects, returns-in-quality)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, PKMixin

if TYPE_CHECKING:
    from db.models.supplier import Supplier


class QualityEvent(Base, PKMixin):
    __tablename__ = "quality_events"

    supplier_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("suppliers.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="SET NULL"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="medium")
    quantity_affected: Mapped[Optional[float]] = mapped_column(Numeric(16, 4), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    supplier: Mapped["Supplier"] = relationship("Supplier", back_populates="quality_events")

    __table_args__ = (
        Index("ix_quality_events_supplier_occurred", "supplier_id", "occurred_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<QualityEvent s={self.supplier_id} type={self.event_type}>"