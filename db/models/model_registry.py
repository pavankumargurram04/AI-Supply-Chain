"""Registry of trained model versions (MLOps source of truth)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from core.enums import ModelStatus
from db.base import Base, PKMixin, TimestampMixin


class ModelRegistry(Base, PKMixin, TimestampMixin):
    __tablename__ = "model_registry"

    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_family: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    dataset_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    feature_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    framework: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    artifact_uri: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    extras: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ModelStatus.TRAINED.value, index=True
    )

    trained_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    validated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    archived_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_registry_name_version"),
        Index("ix_model_registry_family_status", "model_family", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ModelRegistry {self.model_name} {self.version} status={self.status}>"