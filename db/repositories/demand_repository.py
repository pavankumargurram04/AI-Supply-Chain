"""Repository for DemandRecord."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Optional, Sequence

from sqlalchemy import func, select

from db.models.demand import DemandRecord
from db.repositories.base_repository import BaseRepository


class DemandRepository(BaseRepository[DemandRecord]):
    model = DemandRecord

    def list_for_series(
        self,
        product_id: int,
        warehouse_id: int,
        *,
        start: Optional[date] = None,
        end: Optional[date] = None,
        order_asc: bool = True,
    ) -> Sequence[DemandRecord]:
        stmt = select(DemandRecord).where(
            DemandRecord.product_id == product_id,
            DemandRecord.warehouse_id == warehouse_id,
        )
        if start is not None:
            stmt = stmt.where(DemandRecord.demand_date >= start)
        if end is not None:
            stmt = stmt.where(DemandRecord.demand_date <= end)
        if order_asc:
            stmt = stmt.order_by(DemandRecord.demand_date.asc())
        else:
            stmt = stmt.order_by(DemandRecord.demand_date.desc())
        return list(self.session.execute(stmt).scalars().all())

    def total_for_series(
        self,
        product_id: int,
        warehouse_id: int,
        *,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> float:
        stmt = select(func.coalesce(func.sum(DemandRecord.quantity), 0.0)).where(
            DemandRecord.product_id == product_id,
            DemandRecord.warehouse_id == warehouse_id,
        )
        if start is not None:
            stmt = stmt.where(DemandRecord.demand_date >= start)
        if end is not None:
            stmt = stmt.where(DemandRecord.demand_date <= end)
        return float(self.session.execute(stmt).scalar_one())

    def bulk_upsert(self, rows: Iterable[dict]) -> int:
        """Insert-or-update by (product_id, warehouse_id, demand_date).

        Uses PostgreSQL ON CONFLICT via the SQLAlchemy dialect insert.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        rows = list(rows)
        if not rows:
            return 0
        stmt = pg_insert(DemandRecord).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["product_id", "warehouse_id", "demand_date"],
            set_={
                "quantity": stmt.excluded.quantity,
                "unit_price": stmt.excluded.unit_price,
                "promotion_flag": stmt.excluded.promotion_flag,
                "holiday_flag": stmt.excluded.holiday_flag,
                "is_censored": stmt.excluded.is_censored,
                "source": stmt.excluded.source,
            },
        )
        self.session.execute(stmt)
        self.session.flush()
        return len(rows)

    def distinct_series(self) -> Sequence[tuple[int, int]]:
        stmt = select(DemandRecord.product_id, DemandRecord.warehouse_id).distinct()
        return [(r[0], r[1]) for r in self.session.execute(stmt).all()]