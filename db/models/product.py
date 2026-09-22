"""Product master data (SKU catalogue)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.enums import LifecycleStage, SeasonalityClass, VolatilityClass
from db.base import ActiveMixin, Base, PKMixin, TimestampMixin

if TYPE_CHECKING:
    from db.models.category import Category
    from db.models.demand import DemandRecord
    from db.models.inventory import Inventory
    from db.models.supplier_product import SupplierProduct


class Product(Base, PKMixin, TimestampMixin, ActiveMixin):
    __tablename__ = "products"

    sku: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    category_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("categories.id", ondelete="SET NULL"), nullable=True
    )

    # Commercials
    unit_cost: Mapped[float] = mapped_column(Numeric(14, 4), nullable=False)
    unit_price: Mapped[float] = mapped_column(Numeric(14, 4), nullable=False)
    weight_kg: Mapped[Optional[float]] = mapped_column(Numeric(10, 4), nullable=True)
    volume_m3: Mapped[Optional[float]] = mapped_column(Numeric(10, 6), nullable=True)

    # Classification
    lifecycle_stage: Mapped[str] = mapped_column(
        String(32), nullable=False, default=LifecycleStage.MATURITY.value
    )
    seasonality_class: Mapped[str] = mapped_column(
        String(32), nullable=False, default=SeasonalityClass.NONE.value
    )
    volatility_class: Mapped[str] = mapped_column(
        String(32), nullable=False, default=VolatilityClass.MEDIUM.value
    )

    # Planning inputs
    base_demand: Mapped[float] = mapped_column(Numeric(14, 4), nullable=False, default=0.0)
    lead_time_days: Mapped[int] = mapped_column(BigInteger, nullable=False, default=7)

    # Relationships
    category: Mapped[Optional["Category"]] = relationship(
        "Category", back_populates="products"
    )
    inventories: Mapped[list["Inventory"]] = relationship(
        "Inventory", back_populates="product", cascade="all, delete-orphan"
    )
    demand_records: Mapped[list["DemandRecord"]] = relationship(
        "DemandRecord", back_populates="product", cascade="all, delete-orphan"
    )
    supplier_links: Mapped[list["SupplierProduct"]] = relationship(
        "SupplierProduct", back_populates="product", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_products_category_id", "category_id"),
        Index("ix_products_lifecycle_stage", "lifecycle_stage"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Product id={self.id} sku={self.sku!r}>"