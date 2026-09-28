"""Shipment endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db
from core.errors import NotFoundError
from db.models.shipment import Shipment
from db.models.order import PurchaseOrder

router = APIRouter(prefix="/shipments", tags=["shipments"],
                   dependencies=[Depends(get_current_user)])


@router.get("")
def list_shipments(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict]:
    q = (
    db.query(Shipment)
    .join(PurchaseOrder, Shipment.purchase_order_id == PurchaseOrder.id)
    .order_by(Shipment.created_at.desc())
)
    if status:
        q = q.filter(Shipment.status == status)
    rows = q.limit(limit).all()
    return [
        {
            "id": r.id,
            "purchase_order_id": r.purchase_order_id,
            "po_number": r.order.po_number,
            "carrier": r.carrier,
            "tracking_number": r.tracking_number,
            "status": r.status,
            "dispatched_at": r.dispatched_at.isoformat() if r.dispatched_at else None,
            "expected_arrival": r.expected_arrival.isoformat() if r.expected_arrival else None,
            "actual_arrival": r.actual_arrival.isoformat() if r.actual_arrival else None,
        }
        for r in rows
    ]


@router.get("/{shipment_id}")
def get_shipment(shipment_id: int, db: Session = Depends(get_db)) -> dict:
    row = db.get(Shipment, shipment_id)
    if row is None:
        raise NotFoundError(f"Shipment {shipment_id} not found")
    events = [
        {
            "event_type": e.event_type,
            "occurred_at": e.occurred_at.isoformat() if e.occurred_at else None,
            "location": e.location, "notes": e.notes,
        }
        for e in row.events
    ]
    return {
        "id": row.id, "status": row.status,
        "carrier": row.carrier, "tracking_number": row.tracking_number,
        "expected_arrival": row.expected_arrival.isoformat() if row.expected_arrival else None,
        "actual_arrival": row.actual_arrival.isoformat() if row.actual_arrival else None,
        "events": events,
    }