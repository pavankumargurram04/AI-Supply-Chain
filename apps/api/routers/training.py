"""Training job endpoints. Enqueue Celery tasks and inspect status."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db, require_admin
from core.errors import NotFoundError, ValidationError
from db.models.training_job import TrainingJob

router = APIRouter(prefix="/training", tags=["training"])


class EnqueueTrainingIn(BaseModel):
    model_family: str
    mode: str = "dev"
    seed: int = 42
    config: dict | None = None


@router.post("/jobs", dependencies=[Depends(require_admin)])
def enqueue_training(payload: EnqueueTrainingIn, db: Session = Depends(get_db), user = Depends(get_current_user)) -> dict:
    """Enqueue a Celery training task; creates a training_jobs row."""
    if payload.model_family not in ("demand_lstm", "supplier_xgb", "inventory_dqn"):
        raise ValidationError(f"Unknown model_family: {payload.model_family}")

    job = TrainingJob(
        model_family=payload.model_family,
        status="queued",
        seed=payload.seed,
        config=payload.config or {"mode": payload.mode},
        dataset_version=None,
        feature_version=None,
        created_by=user.id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Dispatch to Celery. Import lazily to avoid a hard dependency at import time.
    try:
        from apps.worker.tasks.training_tasks import dispatch_training_job
        dispatch_training_job.delay(job.id)
    except Exception as e:  # noqa: BLE001
        # If the broker is not reachable we still return the queued row —
        # the user can re-dispatch.
        return {"job_id": job.id, "status": job.status, "dispatched": False, "error": str(e)}

    return {"job_id": job.id, "status": job.status, "dispatched": True}


@router.get("/jobs/{job_id}", dependencies=[Depends(get_current_user)])
def get_job(job_id: int, db: Session = Depends(get_db)) -> dict:
    row = db.get(TrainingJob, job_id)
    if row is None:
        raise NotFoundError(f"Training job {job_id} not found")
    return {
        "id": row.id,
        "model_family": row.model_family,
        "status": row.status,
        "seed": row.seed,
        "config": row.config,
        "metrics": row.metrics,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "error_message": row.error_message,
    }


@router.get("/jobs", dependencies=[Depends(get_current_user)])
def list_jobs(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[dict]:
    q = db.query(TrainingJob).order_by(TrainingJob.created_at.desc())
    if status:
        q = q.filter(TrainingJob.status == status)
    rows = q.limit(limit).all()
    return [
        {"id": r.id, "model_family": r.model_family, "status": r.status,
         "seed": r.seed, "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in rows
    ]

