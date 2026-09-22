"""Operational and ML alerts."""

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
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from core.enums import AlertSeverity, AlertStatus
from db.base import Base, PKMixin, TimestampMixin


class Alert(Base, PKMixin, TimestampMixin):
    __tablename__ = "alerts"

    severity: Mapped[str] = mapped_column(
        String(16), nullable=False, default=AlertSeverity.INFO.value, index=True
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=AlertStatus.OPEN.value, index=True
    )

    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    entity_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    entity_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolved_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    __table_args__ = (
        Index("ix_alerts_severity_status", "severity", "status"),
        Index("ix_alerts_source_created", "source", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Alert id={self.id} severity={self.severity} status={self.status}>"