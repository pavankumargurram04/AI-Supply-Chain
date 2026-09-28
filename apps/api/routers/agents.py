"""Multi-agent orchestrator endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db, require_analyst
from apps.api.schemas import AgentMessageOut, AgentRunIn, DecisionTraceOut
from apps.api.services import build_orchestrator
from core.errors import NotFoundError

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/run", dependencies=[Depends(require_analyst)])
def run_agents(payload: AgentRunIn) -> dict:
    """Run the full agent orchestration and return messages + trace."""
    from decision_engine.decision_trace import DecisionRequest

    orch = build_orchestrator(payload.mode if hasattr(payload, "mode") else "dev")
    request = DecisionRequest(
        product_sku=payload.product_sku,
        warehouse_code=payload.warehouse_code,
        service_level=payload.service_level,
        horizon_days=payload.horizon_days,
        demand_multiplier=payload.demand_multiplier,
        lead_time_override_days=payload.lead_time_override_days,
        inventory_override=payload.inventory_override,
        seed=payload.seed,
    )
    result = orch.run(request)

    return {
        "trace_id": result.trace_id,
        "run_at": result.run_at,
        "messages": [
            AgentMessageOut(
                message_id=m.message_id,
                trace_id=m.trace_id,
                agent=m.agent,
                timestamp=m.timestamp,
                status=m.status,
                confidence=m.confidence,
                payload=m.payload,
                warnings=m.warnings,
                errors=m.errors,
                model_versions=m.model_versions,
            ).model_dump()
            for m in result.messages
        ],
        "decision_trace": result.decision_trace.to_dict(),
    }


@router.get("/trace/{trace_id}", dependencies=[Depends(get_current_user)])
def get_trace(trace_id: str, db: Session = Depends(get_db)) -> dict:
    """Fetch a previously persisted trace from recommendation_runs."""
    from db.models.recommendation import RecommendationRun
    row = db.query(RecommendationRun).filter_by(trace_id=trace_id).one_or_none()
    if row is None:
        raise NotFoundError(f"Trace {trace_id} not found")
    return {
        "trace_id": row.trace_id,
        "status": row.status,
        "decision_trace": row.decision_trace,
        "explanation": row.explanation,
        "confidence": row.confidence,
    }