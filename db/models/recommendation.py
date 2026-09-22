"""Procurement recommendation runs and their line items."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.enums import RecommendationStatus
from db.base import Base, PKMixin, TimestampMixin

if TYPE_CHECKING:
    pass


class RecommendationRun(Base, PKMixin, TimestampMixin):
    __tablename__ = "recommendation_runs"

    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    warehouse_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("warehouses.id", ondelete="CASCADE"), nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=RecommendationStatus.GENERATED.value, index=True
    )

    # Full input snapshot for reproducibility.
    request: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    input_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Intermediate artefacts.
    forecast: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    supplier_risks: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    inventory_state: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    dqn_action: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    optimization_result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    decision_trace: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Final numeric outputs.
    stockout_probability: Mapped[Optional[float]] = mapped_column(Numeric(10, 6), nullable=True)
    safety_stock: Mapped[Optional[float]] = mapped_column(Numeric(16, 4), nullable=True)
    reorder_point: Mapped[Optional[float]] = mapped_column(Numeric(16, 4), nullable=True)
    final_quantity: Mapped[Optional[float]] = mapped_column(Numeric(16, 4), nullable=True)
    supplier_allocation: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    expected_cost: Mapped[Optional[float]] = mapped_column(Numeric(16, 4), nullable=True)
    expected_service_level: Mapped[Optional[float]] = mapped_column(Numeric(10, 6), nullable=True)

    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(6, 4), nullable=True)

    # Model provenance.
    demand_model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    supplier_model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    inventory_model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    optimization_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    data_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    feature_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Human workflow.
    created_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    reviewed_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    override: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    items: Mapped[list["RecommendationItem"]] = relationship(
        "RecommendationItem", back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_rec_runs_product_wh", "product_id", "warehouse_id"),
        Index("ix_rec_runs_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RecommendationRun id={self.id} trace={self.trace_id} status={self.status}>"


class RecommendationItem(Base, PKMixin, TimestampMixin):
    __tablename__ = "recommendation_items"

    recommendation_run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("recommendation_runs.id", ondelete="CASCADE"), nullable=False
    )
    supplier_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("suppliers.id", ondelete="RESTRICT"), nullable=False
    )
    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False
    )

    quantity: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    unit_price: Mapped[float] = mapped_column(Numeric(14, 4), nullable=False)
    line_total: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    run: Mapped["RecommendationRun"] = relationship("RecommendationRun", back_populates="items")

    __table_args__ = (
        Index("ix_rec_items_run", "recommendation_run_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RecommendationItem run={self.recommendation_run_id} supplier={self.supplier_id}>"