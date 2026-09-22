"""Supplier master data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.enums import ReliabilityClass
from db.base import ActiveMixin, Base, PKMixin, TimestampMixin

if TYPE_CHECKING:
    from db.models.quality_event import QualityEvent
    from db.models.supplier_event import SupplierEvent
    from db.models.supplier_product import SupplierProduct


class Supplier(Base, PKMixin, TimestampMixin, ActiveMixin):
    __tablename__ = "suppliers"

    code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    contact_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    contact_phone: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    country: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    region: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Classes used by the synthetic generator and as engineered features.
    reliability_class: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ReliabilityClass.STANDARD.value
    )
    cost_class: Mapped[str] = mapped_column(String(32), nullable=False, default="standard")
    quality_class: Mapped[str] = mapped_column(String(32), nullable=False, default="standard")
    lead_time_class: Mapped[str] = mapped_column(String(32), nullable=False, default="standard")
    capacity_class: Mapped[str] = mapped_column(String(32), nullable=False, default="standard")

    contracted_capacity_units: Mapped[Optional[float]] = mapped_column(
        Numeric(16, 4), nullable=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    product_links: Mapped[list["SupplierProduct"]] = relationship(
        "SupplierProduct", back_populates="supplier", cascade="all, delete-orphan"
    )
    performance_events: Mapped[list["SupplierEvent"]] = relationship(
        "SupplierEvent", back_populates="supplier", cascade="all, delete-orphan"
    )
    quality_events: Mapped[list["QualityEvent"]] = relationship(
        "QualityEvent", back_populates="supplier", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_suppliers_reliability_class", "reliability_class"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Supplier id={self.id} code={self.code!r}>"