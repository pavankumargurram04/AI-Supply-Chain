"""Procurement recommendation workflow: recommend, approve, reject, override.

Recommendations are persisted into `recommendation_runs`. Each action is
audit-logged. Override stores both the original recommendation and the
human-modified quantity.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.dependencies import (
    get_current_user,
    get_db,
    require_procurement_manager,
    require_reviewer,
)
from apps.api.schemas import (
    ApproveRequest,
    DecisionRequestIn,
    OverrideRequest,
    RecommendationListItem,
)
from apps.api.services import build_decision_engine
from core.enums import OrderStatus, RecommendationStatus, ShipmentEventType, ShipmentStatus
from core.errors import ConflictError, NotFoundError
from db.models.audit_log import AuditLog
from db.models.order import PurchaseOrder, PurchaseOrderItem
from db.models.product import Product
from db.models.recommendation import RecommendationRun
from db.models.shipment import Shipment
from db.models.shipment_event import ShipmentEvent
from db.models.supplier import Supplier
from db.models.warehouse import Warehouse
from db.repositories.recommendation_repository import RecommendationRepository

router = APIRouter(prefix="/procurement", tags=["procurement"])


# ---------------------------------------------------------------------------
# Recommend (persists a RecommendationRun row)
# ---------------------------------------------------------------------------
@router.post("/recommend", response_model=RecommendationListItem,
             dependencies=[Depends(require_procurement_manager)])
def recommend(
    payload: DecisionRequestIn,
    mode: str = Query(default="dev"),
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
) -> RecommendationListItem:
    """Run the decision engine and persist the recommendation."""
    from decision_engine.decision_trace import DecisionRequest

    engine = build_decision_engine(mode)
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
    )
    trace = engine.run(request)

    # Resolve product/warehouse IDs from natural keys.
    product = db.query(Product).filter_by(sku=payload.product_sku).one_or_none()
    warehouse = db.query(Warehouse).filter_by(code=payload.warehouse_code).one_or_none()
    if product is None or warehouse is None:
        raise NotFoundError(
            "Product or warehouse not found in DB. "
            "Ingest the dataset with scripts/ingest_data.py first."
        )

    run = RecommendationRun(
        trace_id=trace.trace_id,
        product_id=product.id,
        warehouse_id=warehouse.id,
        status=RecommendationStatus.PENDING_REVIEW.value,
        request=trace.request,
        input_snapshot=trace.input_snapshot,
        forecast={
            "point": trace.forecast_point,
            "std": trace.forecast_std,
            "dates": trace.forecast_dates,
            "origin_date": trace.forecast_origin,
        },
        supplier_risks={"rows": trace.supplier_risks},
        inventory_state=trace.inventory_state,
        dqn_action=trace.dqn_action,
        optimization_result=trace.allocation,
        decision_trace=trace.to_dict(),
        stockout_probability=trace.stockout_probability,
        safety_stock=trace.safety_stock,
        reorder_point=trace.reorder_point,
        final_quantity=trace.final_quantity,
        supplier_allocation={"lines": trace.final_suppliers},
        expected_cost=trace.total_cost,
        expected_service_level=trace.expected_service_level,
        explanation=trace.explanation,
        confidence=trace.confidence,
        demand_model_version=trace.demand_model_version,
        supplier_model_version=trace.supplier_model_version,
        inventory_model_version=trace.inventory_model_version,
        optimization_version=trace.optimization_version,
        data_version=trace.data_version,
        feature_version=trace.feature_version,
        created_by=user.id,
    )
    db.add(run)
    db.flush()

    # Audit.
    db.add(AuditLog(
        actor_id=user.id, actor_name=user.username,
        action="recommendation.created",
        entity_type="recommendation_run", entity_id=run.id,
        before=None, after={"status": run.status, "trace_id": run.trace_id},
        trace_id=run.trace_id, notes=None,
    ))
    db.flush()

    return RecommendationListItem.model_validate(run)


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------
@router.get("", response_model=list[RecommendationListItem],
            dependencies=[Depends(get_current_user)])
def list_recommendations(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[RecommendationListItem]:
    repo = RecommendationRepository(db)
    if status:
        rows = repo.list_by_status(status, limit=limit)
    else:
        rows = repo.list(order_by=RecommendationRun.created_at.desc(), limit=limit)
    return [RecommendationListItem.model_validate(r) for r in rows]


@router.get("/{run_id}", response_model=RecommendationListItem,
            dependencies=[Depends(get_current_user)])
def get_recommendation(run_id: int, db: Session = Depends(get_db)) -> RecommendationListItem:
    row = db.get(RecommendationRun, run_id)
    if row is None:
        raise NotFoundError(f"Recommendation {run_id} not found")
    return RecommendationListItem.model_validate(row)


# ---------------------------------------------------------------------------
# Approve / reject / override
# ---------------------------------------------------------------------------
@router.post("/{run_id}/approve", response_model=RecommendationListItem,
             dependencies=[Depends(require_reviewer)])
def approve(
    run_id: int,
    payload: ApproveRequest,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
) -> RecommendationListItem:
    row = db.get(RecommendationRun, run_id)
    if row is None:
        raise NotFoundError(f"Recommendation {run_id} not found")
    if row.status not in (RecommendationStatus.PENDING_REVIEW.value,
                          RecommendationStatus.GENERATED.value):
        raise ConflictError(f"Cannot approve from status {row.status!r}")

    before = {"status": row.status}
    row.status = RecommendationStatus.APPROVED.value
    row.reviewed_by = user.id
    row.reviewed_at = datetime.now(tz=timezone.utc)
    db.flush()

    db.add(AuditLog(
        actor_id=user.id, actor_name=user.username,
        action="recommendation.approved",
        entity_type="recommendation_run", entity_id=row.id,
        before=before, after={"status": row.status, "notes": payload.notes},
        trace_id=row.trace_id,
    ))
    db.flush()
    return RecommendationListItem.model_validate(row)


@router.post("/{run_id}/reject", response_model=RecommendationListItem,
             dependencies=[Depends(require_reviewer)])
def reject(
    run_id: int,
    payload: ApproveRequest,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
) -> RecommendationListItem:
    row = db.get(RecommendationRun, run_id)
    if row is None:
        raise NotFoundError(f"Recommendation {run_id} not found")
    if row.status not in (RecommendationStatus.PENDING_REVIEW.value,
                          RecommendationStatus.GENERATED.value):
        raise ConflictError(f"Cannot reject from status {row.status!r}")

    before = {"status": row.status}
    row.status = RecommendationStatus.REJECTED.value
    row.reviewed_by = user.id
    row.reviewed_at = datetime.now(tz=timezone.utc)
    db.flush()

    db.add(AuditLog(
        actor_id=user.id, actor_name=user.username,
        action="recommendation.rejected",
        entity_type="recommendation_run", entity_id=row.id,
        before=before, after={"status": row.status, "notes": payload.notes},
        trace_id=row.trace_id,
    ))
    db.flush()
    return RecommendationListItem.model_validate(row)


@router.post("/{run_id}/override", response_model=RecommendationListItem,
             dependencies=[Depends(require_procurement_manager)])
def override(
    run_id: int,
    payload: OverrideRequest,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
) -> RecommendationListItem:
    """Human override. Preserves the AI recommendation and the override."""
    row = db.get(RecommendationRun, run_id)
    if row is None:
        raise NotFoundError(f"Recommendation {run_id} not found")
    if row.status in (RecommendationStatus.ORDER_CREATED.value,
                      RecommendationStatus.REJECTED.value):
        raise ConflictError(f"Cannot override from status {row.status!r}")

    before = {
        "status": row.status,
        "final_quantity": float(row.final_quantity or 0.0),
        "override": row.override,
    }
    original_quantity = float(row.final_quantity or 0.0)
    row.override = {
        "original_quantity": original_quantity,
        "new_quantity": float(payload.new_quantity),
        "reason": payload.reason,
        "by_user_id": user.id,
        "at": datetime.now(tz=timezone.utc).isoformat(),
    }
    row.final_quantity = float(payload.new_quantity)
    row.status = RecommendationStatus.OVERRIDDEN.value
    row.reviewed_by = user.id
    row.reviewed_at = datetime.now(tz=timezone.utc)
    db.flush()

    db.add(AuditLog(
        actor_id=user.id, actor_name=user.username,
        action="recommendation.overridden",
        entity_type="recommendation_run", entity_id=row.id,
        before=before,
        after={"status": row.status, "override": row.override},
        trace_id=row.trace_id,
    ))
    db.flush()
    return RecommendationListItem.model_validate(row)


@router.post("/{run_id}/create-order", dependencies=[Depends(require_procurement_manager)])
def create_order_from_recommendation(
    run_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
) -> dict:
    """Create a purchase order (and an initial shipment) from an approved recommendation.

    Preconditions: the recommendation must have status APPROVED or OVERRIDDEN.
    Uses the `final_suppliers` list stored in `supplier_allocation` to build
    the PO items.
    """
    row = db.get(RecommendationRun, run_id)
    if row is None:
        raise NotFoundError(f"Recommendation {run_id} not found")
    if row.status not in (
        RecommendationStatus.APPROVED.value,
        RecommendationStatus.OVERRIDDEN.value,
    ):
        raise ConflictError(
            f"Cannot create PO from status {row.status!r}; approve or override first"
        )

    # Look up the product and warehouse for the trace.
    product = db.get(Product, row.product_id)
    warehouse = db.get(Warehouse, row.warehouse_id)
    if product is None or warehouse is None:
        raise NotFoundError("Product or warehouse not found for this recommendation")

    allocation = row.supplier_allocation or {}
    lines = allocation.get("lines", []) if isinstance(allocation, dict) else []
    if not lines:
        raise ConflictError("Recommendation has no supplier allocation lines")

    # Resolve each supplier_code → supplier_id.
    code_to_id: dict[str, int] = {}
    for line in lines:
        code = line.get("supplier_code")
        if not code or code in code_to_id:
            continue
        sup = db.query(Supplier).filter_by(code=code).one_or_none()
        if sup is None:
            raise NotFoundError(f"Supplier {code} not found in database")
        code_to_id[code] = sup.id

    # Determine a natural expected delivery date: max lead time across lines.
    max_lt = max(float(l.get("lead_time_days", 7.0)) for l in lines)
    expected = date.today() + timedelta(days=int(round(max_lt)))

    # Prefer a single PO per recommendation (group by primary supplier).
    po_number = f"PO-{row.trace_id[:10].upper()}"
    po = PurchaseOrder(
        po_number=po_number,
        supplier_id=next(iter(code_to_id.values())),
        warehouse_id=warehouse.id,
        status=OrderStatus.ORDERED.value,
        expected_delivery_date=expected,
        total_amount=float(row.expected_cost or 0.0),
        recommendation_id=row.id,
        created_by=user.id,
        approved_by=user.id,
        approved_at=datetime.now(tz=timezone.utc),
    )
    db.add(po)
    db.flush()

    total_amount = 0.0
    for line in lines:
        supplier_code = line.get("supplier_code")
        supplier_id = code_to_id.get(supplier_code)
        if supplier_id is None:
            continue
        qty = float(line.get("quantity", 0.0))
        unit_price = float(line.get("unit_price", 0.0))
        line_total = float(line.get("line_total", qty * unit_price))
        db.add(PurchaseOrderItem(
            purchase_order_id=po.id,
            product_id=product.id,
            quantity=qty,
            unit_price=unit_price,
            line_total=line_total,
        ))
        total_amount += line_total

    po.total_amount = total_amount

    # Create an initial shipment in PLANNED state.
    shipment = Shipment(
        purchase_order_id=po.id,
        carrier="Pending assignment",
        tracking_number=None,
        status=ShipmentStatus.PLANNED.value,
        expected_arrival=expected,
    )
    db.add(shipment)
    db.flush()

    db.add(ShipmentEvent(
        shipment_id=shipment.id,
        event_type=ShipmentEventType.CREATED.value,
        notes=f"Created from recommendation {row.trace_id}",
    ))

    # Move recommendation forward.
    before_status = row.status
    row.status = RecommendationStatus.ORDER_CREATED.value

    # Audit.
    db.add(AuditLog(
        actor_id=user.id, actor_name=user.username,
        action="recommendation.order_created",
        entity_type="recommendation_run", entity_id=row.id,
        before={"status": before_status},
        after={
            "status": row.status,
            "po_id": po.id,
            "po_number": po.po_number,
            "shipment_id": shipment.id,
            "total_amount": total_amount,
        },
        trace_id=row.trace_id,
    ))
    db.flush()

    return {
        "purchase_order_id": po.id,
        "po_number": po.po_number,
        "shipment_id": shipment.id,
        "total_amount": total_amount,
        "expected_delivery_date": expected.isoformat(),
        "n_items": len(lines),
    }