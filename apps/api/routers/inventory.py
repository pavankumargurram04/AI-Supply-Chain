"""Inventory intelligence endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db
from apps.api.schemas import DecisionRequestIn, DecisionTraceOut
from apps.api.services import build_decision_engine
from core.errors import ValidationError

router = APIRouter(prefix="/inventory", tags=["inventory"],
                   dependencies=[Depends(get_current_user)])


@router.get("/{product_sku}/{warehouse_code}")
def get_inventory_state(
    product_sku: str,
    warehouse_code: str,
    mode: str = Query(default="dev"),
    db: Session = Depends(get_db),
) -> dict:
    """Return the current on-hand/reserved/inbound snapshot for one series."""
    from decision_engine.state_builder import build_state

    from apps.api.services import curated_features, raw_dataset

    state = build_state(
        product_sku=product_sku,
        warehouse_code=warehouse_code,
        curated_features_dir=curated_features(mode),
        raw_dir=raw_dataset(mode),
    )
    return {
        "product_sku": state.product_sku,
        "warehouse_code": state.warehouse_code,
        "on_hand": state.on_hand,
        "reserved": state.reserved,
        "inbound": state.inbound,
        "available": float(state.on_hand - state.reserved),
        "inventory_position": float(state.on_hand - state.reserved + state.inbound),
    }


@router.post("/recommend", response_model=DecisionTraceOut)
def recommend(payload: DecisionRequestIn, mode: str = Query(default="dev")) -> DecisionTraceOut:
    """Run the full decision engine for one series."""
    engine = build_decision_engine(mode)
    from decision_engine.decision_trace import DecisionRequest

    request = DecisionRequest(
        product_sku=payload.product_sku,
        warehouse_code=payload.warehouse_code,
        service_level=payload.service_level,
        horizon_days=payload.horizon_days,
        demand_multiplier=payload.demand_multiplier,
        lead_time_override_days=payload.lead_time_override_days,
        lead_time_std_override=payload.lead_time_std_override,
        inventory_override=payload.inventory_override,
        budget=payload.budget,
        max_suppliers=payload.max_suppliers,
        preferred_supplier_codes=payload.preferred_supplier_codes,
        seed=payload.seed,
        scenario_id=payload.scenario_id,
    )
    trace = engine.run(request)
    return DecisionTraceOut(
        trace_id=trace.trace_id,
        run_at=trace.run_at,
        product_sku=payload.product_sku,
        warehouse_code=payload.warehouse_code,
        final_quantity=trace.final_quantity,
        total_cost=trace.total_cost,
        expected_service_level=trace.expected_service_level,
        stockout_probability=trace.stockout_probability,
        safety_stock=trace.safety_stock,
        reorder_point=trace.reorder_point,
        confidence=trace.confidence,
        guardrail_violations=trace.guardrail_violations,
        explanation=trace.explanation,
        full_trace=trace.to_dict(),
    )