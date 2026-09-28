"""Purchase order endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db
from db.models.order import PurchaseOrder

router = APIRouter(
    prefix="/orders",
    tags=["orders"],
    dependencies=[Depends(get_current_user)],
)


@router.get("")
def list_orders(
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = db.query(PurchaseOrder).order_by(PurchaseOrder.created_at.desc())

    if status:
        query = query.filter(PurchaseOrder.status == status)

    rows = query.limit(limit).all()

    return [
        {
            "id": row.id,
            "po_number": row.po_number,
            "supplier_id": row.supplier_id,
            "warehouse_id": row.warehouse_id,
            "status": row.status,
            "expected_delivery_date": (
                row.expected_delivery_date.isoformat()
                if row.expected_delivery_date
                else None
            ),
            "actual_delivery_date": (
                row.actual_delivery_date.isoformat()
                if row.actual_delivery_date
                else None
            ),
            "total_amount": float(row.total_amount),
            "currency": row.currency,
            "recommendation_id": row.recommendation_id,
            "created_at": (
                row.created_at.isoformat()
                if row.created_at
                else None
            ),
        }
        for row in rows
    ]


@router.get("/{order_id}")
def get_order(
    order_id: int,
    db: Session = Depends(get_db),
) -> dict:
    row = db.get(PurchaseOrder, order_id)

    if row is None:
        from core.errors import NotFoundError

        raise NotFoundError(f"Purchase order {order_id} not found")

    items = [
        {
            "id": item.id,
            "product_id": item.product_id,
            "quantity": float(item.quantity),
            "received_quantity": float(item.received_quantity),
            "unit_price": float(item.unit_price),
            "line_total": float(item.line_total),
        }
        for item in row.items
    ]

    shipments = [
        {
            "id": shipment.id,
            "carrier": shipment.carrier,
            "tracking_number": shipment.tracking_number,
            "status": shipment.status,
            "expected_arrival": (
                shipment.expected_arrival.isoformat()
                if shipment.expected_arrival
                else None
            ),
            "actual_arrival": (
                shipment.actual_arrival.isoformat()
                if shipment.actual_arrival
                else None
            ),
        }
        for shipment in row.shipments
    ]

    return {
        "id": row.id,
        "po_number": row.po_number,
        "supplier_id": row.supplier_id,
        "warehouse_id": row.warehouse_id,
        "status": row.status,
        "expected_delivery_date": (
            row.expected_delivery_date.isoformat()
            if row.expected_delivery_date
            else None
        ),
        "actual_delivery_date": (
            row.actual_delivery_date.isoformat()
            if row.actual_delivery_date
            else None
        ),
        "total_amount": float(row.total_amount),
        "currency": row.currency,
        "recommendation_id": row.recommendation_id,
        "created_at": (
            row.created_at.isoformat()
            if row.created_at
            else None
        ),
        "notes": row.notes,
        "items": items,
        "shipments": shipments,
    }