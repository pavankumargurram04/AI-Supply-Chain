"""Warehouse / distribution centre master data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import ActiveMixin, Base, PKMixin, TimestampMixin

if TYPE_CHECKING:
    from db.models.inventory import Inventory


class Warehouse(Base, PKMixin, TimestampMixin, ActiveMixin):
    __tablename__ = "warehouses"

    code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    region: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    country: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Numeric(9, 6), nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Numeric(9, 6), nullable=True)

    capacity_units: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False, default=0.0)
    operating_cost_daily: Mapped[float] = mapped_column(
        Numeric(14, 4), nullable=False, default=0.0
    )

    inventories: Mapped[list["Inventory"]] = relationship(
        "Inventory", back_populates="warehouse", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_warehouses_region", "region"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Warehouse id={self.id} code={self.code!r}>"