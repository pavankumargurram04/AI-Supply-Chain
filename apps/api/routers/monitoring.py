"""Monitoring endpoints: alerts, drift, usage."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db
from db.models.alert import Alert
from monitoring.usage_metrics import snapshot_usage

router = APIRouter(prefix="/monitoring", tags=["monitoring"],
                   dependencies=[Depends(get_current_user)])


@router.get("/alerts")
def list_alerts(
    status: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict]:
    q = db.query(Alert).order_by(Alert.created_at.desc())
    if status:
        q = q.filter(Alert.status == status)
    if severity:
        q = q.filter(Alert.severity == severity)
    rows = q.limit(limit).all()
    return [
        {
            "id": r.id, "severity": r.severity, "status": r.status,
            "source": r.source, "title": r.title, "message": r.message,
            "entity_type": r.entity_type, "entity_id": r.entity_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "payload": r.payload,
        }
        for r in rows
    ]


@router.get("/usage")
def usage(days: int = Query(default=7, ge=1, le=90), db: Session = Depends(get_db)) -> dict:
    return snapshot_usage(db, days=days)