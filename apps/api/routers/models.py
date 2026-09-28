"""Model registry endpoints: list, inspect, promote, rollback."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db, require_admin
from core.errors import ConflictError, NotFoundError
from model_registry.promotion import (
    MetricThreshold,
    PromotionCriteria,
    evaluate_promotion,
    promote,
)
from model_registry.registry import Registry
from model_registry.rollback import rollback as do_rollback

router = APIRouter(prefix="/models", tags=["models"])


class PromoteIn(BaseModel):
    thresholds: list[dict] = []
    primary_metric_path: Optional[str] = "test.wape"
    primary_metric_direction: str = "min"
    min_relative_improvement: float = 0.0


@router.get("", dependencies=[Depends(get_current_user)])
def list_models(
    family: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> list[dict]:
    reg = Registry(db)
    if family:
        rows = reg.list_for_family(family)
    elif status:
        rows = reg.list_by_status(status)
    else:
        from db.models.model_registry import ModelRegistry
        rows = db.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "model_name": r.model_name,
            "family": r.model_family,
            "version": r.version,
            "status": r.status,
            "dataset_version": r.dataset_version,
            "feature_version": r.feature_version,
            "metrics": r.metrics,
            "trained_at": r.trained_at.isoformat() if r.trained_at else None,
            "promoted_at": r.promoted_at.isoformat() if r.promoted_at else None,
        }
        for r in rows
    ]


@router.get("/{model_name}/{version}", dependencies=[Depends(get_current_user)])
def get_model(model_name: str, version: str, db: Session = Depends(get_db)) -> dict:
    reg = Registry(db)
    row = reg.get(model_name=model_name, version=version)
    if row is None:
        raise NotFoundError(f"{model_name} {version} not found")
    return {
        "model_name": row.model_name,
        "version": row.version,
        "status": row.status,
        "hyperparameters": row.hyperparameters,
        "metrics": row.metrics,
        "artifact_uri": row.artifact_uri,
        "extras": row.extras,
    }


@router.post("/{model_name}/{version}/promote", dependencies=[Depends(require_admin)])
def promote_endpoint(
    model_name: str,
    version: str,
    payload: PromoteIn,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
) -> dict:
    reg = Registry(db)
    # Move through stages: validated → staging → (promote handles production).
    row = reg.get(model_name=model_name, version=version)
    if row is None:
        raise NotFoundError(f"{model_name} {version} not found")
    if row.status == "trained":
        reg.mark_validated(model_name=model_name, version=version)
        row = reg.get(model_name=model_name, version=version)
    if row.status == "validated":
        reg.mark_staging(model_name=model_name, version=version)

    criteria = PromotionCriteria(
        thresholds=[MetricThreshold(**t) for t in payload.thresholds],
        primary_metric_path=payload.primary_metric_path,
        primary_metric_direction=payload.primary_metric_direction,
        min_relative_improvement=payload.min_relative_improvement,
    )
    result = promote(reg, model_name=model_name, version=version,
                     criteria=criteria, actor=user.username)
    if not result.ok:
        raise ConflictError("Promotion rejected", details=result.to_dict())
    return result.to_dict()


@router.post("/{model_name}/rollback", dependencies=[Depends(require_admin)])
def rollback_endpoint(
    model_name: str,
    to_version: Optional[str] = Query(default=None),
    reason: str = Query(default="manual rollback"),
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
) -> dict:
    reg = Registry(db)
    result = do_rollback(reg, model_name=model_name, to_version=to_version,
                         actor=user.username, reason=reason)
    if not result.ok:
        raise ConflictError(result.reason)
    return result.to_dict()