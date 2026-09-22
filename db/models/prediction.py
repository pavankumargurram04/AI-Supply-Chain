"""Stored model predictions (provenance for every numeric output)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, PKMixin


class Prediction(Base, PKMixin):
    __tablename__ = "prediction_records"

    model_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("model_registry.id", ondelete="SET NULL"), nullable=True
    )
    model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    prediction_type: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )  # e.g. "demand", "supplier_risk"
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    product_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    supplier_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    warehouse_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    data_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    feature_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    input_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    uncertainty: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_pred_type_entity", "prediction_type", "entity_type", "entity_id"),
        Index("ix_pred_product_created", "product_id", "created_at"),
        Index("ix_pred_supplier_created", "supplier_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Prediction id={self.id} type={self.prediction_type} "
            f"entity={self.entity_type}:{self.entity_id}>"
        )