"""Training job lifecycle records."""

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
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from core.enums import TrainingJobStatus
from db.base import Base, PKMixin, TimestampMixin


class TrainingJob(Base, PKMixin, TimestampMixin):
    __tablename__ = "training_jobs"

    model_family: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=TrainingJobStatus.QUEUED.value, index=True
    )

    dataset_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    feature_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    seed: Mapped[int] = mapped_column(BigInteger, nullable=False, default=42)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    result_model_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("model_registry.id", ondelete="SET NULL"), nullable=True
    )
    logs_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    __table_args__ = (
        Index("ix_training_jobs_family_status", "model_family", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TrainingJob id={self.id} family={self.model_family} status={self.status}>"