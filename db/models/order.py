"""Purchase orders and their line items.

Two tables in one module: `PurchaseOrder` and `PurchaseOrderItem`.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.enums import OrderStatus
from db.base import Base, PKMixin, TimestampMixin

if TYPE_CHECKING:
    from db.models.shipment import Shipment
    from db.models.supplier import Supplier
    from db.models.warehouse import Warehouse


class PurchaseOrder(Base, PKMixin, TimestampMixin):
    __tablename__ = "purchase_orders"

    po_number: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    supplier_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("suppliers.id", ondelete="RESTRICT"), nullable=False
    )
    warehouse_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("warehouses.id", ondelete="RESTRICT"), nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=OrderStatus.DRAFT.value, index=True
    )

    expected_delivery_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    actual_delivery_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    total_amount: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False, default=0.0)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")

    # Link to the recommendation that created this PO (if any).
    recommendation_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("recommendation_runs.id", ondelete="SET NULL"), nullable=True
    )

    created_by: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    approved_by: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    supplier: Mapped["Supplier"] = relationship("Supplier", foreign_keys=[supplier_id])
    warehouse: Mapped["Warehouse"] = relationship("Warehouse", foreign_keys=[warehouse_id])
    items: Mapped[list["PurchaseOrderItem"]] = relationship(
        "PurchaseOrderItem", back_populates="order", cascade="all, delete-orphan"
    )
    shipments: Mapped[list["Shipment"]] = relationship(
        "Shipment", back_populates="order", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_po_supplier_status", "supplier_id", "status"),
        Index("ix_po_warehouse", "warehouse_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PurchaseOrder id={self.id} po={self.po_number!r} status={self.status}>"


class PurchaseOrderItem(Base, PKMixin, TimestampMixin):
    __tablename__ = "purchase_order_items"

    purchase_order_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("purchase_orders.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False
    )

    quantity: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    received_quantity: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False, default=0.0)
    unit_price: Mapped[float] = mapped_column(Numeric(14, 4), nullable=False)
    line_total: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)

    order: Mapped["PurchaseOrder"] = relationship("PurchaseOrder", back_populates="items")

    __table_args__ = (
        UniqueConstraint(
            "purchase_order_id", "product_id",
            name="uq_po_items_order_product",
        ),
        Index("ix_po_items_product", "product_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PurchaseOrderItem po={self.purchase_order_id} p={self.product_id} qty={self.quantity}>"