"""Shipments attached to a purchase order."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.enums import ShipmentStatus
from db.base import Base, PKMixin, TimestampMixin

if TYPE_CHECKING:
    from db.models.order import PurchaseOrder
    from db.models.shipment_event import ShipmentEvent


class Shipment(Base, PKMixin, TimestampMixin):
    __tablename__ = "shipments"

    purchase_order_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("purchase_orders.id", ondelete="CASCADE"), nullable=False
    )

    carrier: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    tracking_number: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ShipmentStatus.PLANNED.value, index=True
    )

    dispatched_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expected_arrival: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    actual_arrival: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    order: Mapped["PurchaseOrder"] = relationship("PurchaseOrder", back_populates="shipments")
    events: Mapped[list["ShipmentEvent"]] = relationship(
        "ShipmentEvent", back_populates="shipment", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_shipment_po_status", "purchase_order_id", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Shipment id={self.id} status={self.status}>"