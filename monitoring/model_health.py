"""Model health: staleness, missing artifacts, latest training status.

Reads the model_registry and, when a DB session is available, reports:
    - time since last training
    - time since last production promotion
    - whether the artifact directory still exists
    - training job status summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from core.enums import ModelStatus
from core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelHealth:
    model_name: str
    family: str
    version: str
    status: str
    trained_at: Optional[str]
    promoted_at: Optional[str]
    artifact_exists: bool
    days_since_trained: Optional[int]
    days_since_promoted: Optional[int]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "family": self.family,
            "version": self.version,
            "status": self.status,
            "trained_at": self.trained_at,
            "promoted_at": self.promoted_at,
            "artifact_exists": self.artifact_exists,
            "days_since_trained": self.days_since_trained,
            "days_since_promoted": self.days_since_promoted,
            "notes": list(self.notes),
        }


def _days_between(then: Optional[datetime], now: datetime) -> Optional[int]:
    if then is None:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return int((now - then).days)


def snapshot_health(session: Session, *, stale_days: int = 180) -> List[ModelHealth]:
    from db.models.model_registry import ModelRegistry

    now = datetime.now(tz=timezone.utc)
    rows = session.query(ModelRegistry).all()

    out: List[ModelHealth] = []
    for r in rows:
        artifact_exists = False
        if r.artifact_path:
            artifact_exists = Path(r.artifact_path).exists()
        h = ModelHealth(
            model_name=r.model_name,
            family=r.model_family,
            version=r.version,
            status=r.status,
            trained_at=r.trained_at.isoformat() if r.trained_at else None,
            promoted_at=r.promoted_at.isoformat() if r.promoted_at else None,
            artifact_exists=artifact_exists,
            days_since_trained=_days_between(r.trained_at, now),
            days_since_promoted=_days_between(r.promoted_at, now),
        )
        if h.days_since_trained is not None and h.days_since_trained > stale_days:
            h.notes.append(f"last trained {h.days_since_trained} days ago (stale > {stale_days})")
        if r.status == ModelStatus.PRODUCTION.value and not artifact_exists:
            h.notes.append("production model artifact missing")
        if r.status == ModelStatus.PRODUCTION.value and r.promoted_at is None:
            h.notes.append("production model has no promoted_at timestamp")
        out.append(h)

    return out