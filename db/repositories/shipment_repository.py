"""Repository for Shipment and ShipmentEvent."""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select

from db.models.shipment import Shipment
from db.models.shipment_event import ShipmentEvent
from db.repositories.base_repository import BaseRepository


class ShipmentRepository(BaseRepository[Shipment]):
    model = Shipment

    def list_by_status(
        self, status: str, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Shipment]:
        return self.list(
            filters={"status": status},
            order_by=Shipment.created_at.desc(),
            limit=limit,
            offset=offset,
        )

    def list_for_order(self, purchase_order_id: int) -> Sequence[Shipment]:
        stmt = (
            select(Shipment)
            .where(Shipment.purchase_order_id == purchase_order_id)
            .order_by(Shipment.id.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_tracking(self, tracking_number: str) -> Optional[Shipment]:
        return self.get_by(tracking_number=tracking_number)

    def list_events(self, shipment_id: int) -> Sequence[ShipmentEvent]:
        stmt = (
            select(ShipmentEvent)
            .where(ShipmentEvent.shipment_id == shipment_id)
            .order_by(ShipmentEvent.occurred_at.asc())
        )
        return list(self.session.execute(stmt).scalars().all())