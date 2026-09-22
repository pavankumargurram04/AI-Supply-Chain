"""Scenario Lab runs (baseline vs. scenario comparison)."""

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
)
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, PKMixin, TimestampMixin


class ScenarioRun(Base, PKMixin, TimestampMixin):
    __tablename__ = "scenario_runs"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    scenario_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")

    seed: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    baseline_result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    scenario_result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    delta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_scenario_type_created", "scenario_type", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ScenarioRun id={self.id} type={self.scenario_type} status={self.status}>"