"""Audit trail endpoint. Read-only view over audit_logs."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db
from db.models.audit_log import AuditLog

router = APIRouter(prefix="/audit", tags=["audit"],
                   dependencies=[Depends(get_current_user)])


@router.get("")
def list_audit(
    entity_type: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    trace_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict]:
    q = db.query(AuditLog).order_by(AuditLog.created_at.desc())
    if entity_type:
        q = q.filter(AuditLog.entity_type == entity_type)
    if action:
        q = q.filter(AuditLog.action == action)
    if trace_id:
        q = q.filter(AuditLog.trace_id == trace_id)
    rows = q.limit(limit).all()
    return [
        {
            "id": r.id,
            "actor_id": r.actor_id,
            "actor_name": r.actor_name,
            "action": r.action,
            "entity_type": r.entity_type,
            "entity_id": r.entity_id,
            "before": r.before,
            "after": r.after,
            "request_id": r.request_id,
            "trace_id": r.trace_id,
            "notes": r.notes,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]