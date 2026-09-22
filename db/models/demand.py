"""Daily observed demand per product / warehouse."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, Boolean, Date, ForeignKey, Index, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, PKMixin, TimestampMixin

if TYPE_CHECKING:
    from db.models.product import Product


class DemandRecord(Base, PKMixin, TimestampMixin):
    __tablename__ = "demand_records"

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    warehouse_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("warehouses.id", ondelete="CASCADE"), nullable=False
    )

    demand_date: Mapped[date] = mapped_column(Date, nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False, default=0.0)
    unit_price: Mapped[Optional[float]] = mapped_column(Numeric(14, 4), nullable=True)

    promotion_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    holiday_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Stockout-censored demand should be flagged so training can correct for it.
    is_censored: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # 'observed' (from operations) vs 'synthetic' (generated benchmark).
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="synthetic")

    product: Mapped["Product"] = relationship("Product", back_populates="demand_records")

    __table_args__ = (
        UniqueConstraint(
            "product_id", "warehouse_id", "demand_date",
            name="uq_demand_product_wh_date",
        ),
        Index("ix_demand_date", "demand_date"),
        Index("ix_demand_product_wh_date", "product_id", "warehouse_id", "demand_date"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DemandRecord p={self.product_id} w={self.warehouse_id} "
            f"{self.demand_date} qty={self.quantity}>"
        )