"""API + decision usage metrics.

Counts from `recommendation_runs`, `prediction_records`, and `audit_logs`
over the last N days. Used by the dashboard for operational visibility.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.enums import RecommendationStatus
from core.logging import get_logger

log = get_logger(__name__)


def snapshot_usage(session: Session, *, days: int = 7) -> Dict[str, Any]:
    """Return key usage counts over the last `days` days."""
    from db.models.audit_log import AuditLog
    from db.models.prediction import Prediction
    from db.models.recommendation import RecommendationRun

    since = datetime.now(tz=timezone.utc) - timedelta(days=days)

    def _count(q):
        return int(session.execute(q).scalar_one() or 0)

    recs_total = _count(
        select(func.count()).select_from(RecommendationRun)
        .where(RecommendationRun.created_at >= since)
    )
    recs_by_status = {}
    for status in (RecommendationStatus.GENERATED.value,
                   RecommendationStatus.PENDING_REVIEW.value,
                   RecommendationStatus.APPROVED.value,
                   RecommendationStatus.REJECTED.value,
                   RecommendationStatus.OVERRIDDEN.value,
                   RecommendationStatus.ORDER_CREATED.value):
        recs_by_status[status] = _count(
            select(func.count()).select_from(RecommendationRun)
            .where(RecommendationRun.created_at >= since,
                   RecommendationRun.status == status)
        )

    preds_total = _count(
        select(func.count()).select_from(Prediction)
        .where(Prediction.created_at >= since)
    )
    audit_total = _count(
        select(func.count()).select_from(AuditLog)
        .where(AuditLog.created_at >= since)
    )

    # Approval funnel.
    approvals = recs_by_status.get(RecommendationStatus.APPROVED.value, 0) \
        + recs_by_status.get(RecommendationStatus.ORDER_CREATED.value, 0)
    approvals += recs_by_status.get(RecommendationStatus.OVERRIDDEN.value, 0)
    funnel = {
        "generated": recs_total,
        "reviewed": recs_by_status.get(RecommendationStatus.PENDING_REVIEW.value, 0)
                    + approvals,
        "approved_or_overridden": approvals,
        "rejected": recs_by_status.get(RecommendationStatus.REJECTED.value, 0),
        "order_created": recs_by_status.get(RecommendationStatus.ORDER_CREATED.value, 0),
    }

    return {
        "window_days": days,
        "recommendations": {"total": recs_total, "by_status": recs_by_status},
        "predictions": {"total": preds_total},
        "audit_events": {"total": audit_total},
        "approval_funnel": funnel,
    }