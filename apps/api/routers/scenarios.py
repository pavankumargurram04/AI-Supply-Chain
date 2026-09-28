"""Scenario Lab: run baseline + scenario and produce a delta."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db, require_analyst
from apps.api.schemas import DecisionTraceOut
from apps.api.services import build_decision_engine
from db.models.scenario import ScenarioRun

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


class ScenarioIn(BaseModel):
    product_sku: str
    warehouse_code: str
    mode: str = "dev"
    scenario_type: str = "demand_spike"
    demand_multiplier: float = Field(default=1.0, ge=0.0)
    lead_time_override_days: float | None = None
    inventory_override: float | None = None
    budget: float | None = None
    seed: int = 42


@router.post("/run", dependencies=[Depends(require_analyst)])
def run_scenario(payload: ScenarioIn, db: Session = Depends(get_db)) -> dict:
    """Run baseline + scenario, return both traces and their delta."""
    from decision_engine.decision_trace import DecisionRequest

    engine = build_decision_engine(payload.mode)

    baseline_req = DecisionRequest(
        product_sku=payload.product_sku,
        warehouse_code=payload.warehouse_code,
        seed=payload.seed,
    )
    scenario_req = DecisionRequest(
        product_sku=payload.product_sku,
        warehouse_code=payload.warehouse_code,
        demand_multiplier=payload.demand_multiplier,
        lead_time_override_days=payload.lead_time_override_days,
        inventory_override=payload.inventory_override,
        budget=payload.budget,
        seed=payload.seed,
    )

    baseline = engine.run(baseline_req)
    scenario = engine.run(scenario_req)

    delta = {
        "final_quantity": scenario.final_quantity - baseline.final_quantity,
        "total_cost": scenario.total_cost - baseline.total_cost,
        "stockout_probability": scenario.stockout_probability - baseline.stockout_probability,
        "safety_stock": scenario.safety_stock - baseline.safety_stock,
        "reorder_point": scenario.reorder_point - baseline.reorder_point,
        "expected_service_level": scenario.expected_service_level - baseline.expected_service_level,
    }

    # Persist.
    row = ScenarioRun(
        name=f"{payload.scenario_type}:{payload.product_sku}/{payload.warehouse_code}",
        scenario_type=payload.scenario_type,
        status="completed",
        seed=payload.seed,
        config=payload.model_dump(),
        baseline_result=baseline.to_dict(),
        scenario_result=scenario.to_dict(),
        delta=delta,
    )
    db.add(row)
    db.flush()

    return {
        "scenario_id": row.id,
        "baseline": {
            "final_quantity": baseline.final_quantity,
            "total_cost": baseline.total_cost,
            "stockout_probability": baseline.stockout_probability,
            "safety_stock": baseline.safety_stock,
            "reorder_point": baseline.reorder_point,
            "expected_service_level": baseline.expected_service_level,
        },
        "scenario": {
            "final_quantity": scenario.final_quantity,
            "total_cost": scenario.total_cost,
            "stockout_probability": scenario.stockout_probability,
            "safety_stock": scenario.safety_stock,
            "reorder_point": scenario.reorder_point,
            "expected_service_level": scenario.expected_service_level,
        },
        "delta": delta,
    }


@router.get("/{scenario_id}", dependencies=[Depends(get_current_user)])
def get_scenario(scenario_id: int, db: Session = Depends(get_db)) -> dict:
    row = db.get(ScenarioRun, scenario_id)
    if row is None:
        from core.errors import NotFoundError
        raise NotFoundError(f"Scenario {scenario_id} not found")
    return {
        "id": row.id,
        "name": row.name,
        "scenario_type": row.scenario_type,
        "status": row.status,
        "config": row.config,
        "baseline_result": row.baseline_result,
        "scenario_result": row.scenario_result,
        "delta": row.delta,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }