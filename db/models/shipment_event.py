"""Append-only log of shipment lifecycle events."""

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
    from db.models.shipment import Shipment


class ShipmentEvent(Base, PKMixin):
    __tablename__ = "shipment_events"

    shipment_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("shipments.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    location: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    shipment: Mapped["Shipment"] = relationship("Shipment", back_populates="events")

    __table_args__ = (
        Index("ix_shipment_events_shipment_occurred", "shipment_id", "occurred_at"),
        Index("ix_shipment_events_type", "event_type"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ShipmentEvent s={self.shipment_id} type={self.event_type}>"