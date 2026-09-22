"""Repository for RecommendationRun and RecommendationItem."""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select

from db.models.recommendation import RecommendationItem, RecommendationRun
from db.repositories.base_repository import BaseRepository


class RecommendationRepository(BaseRepository[RecommendationRun]):
    model = RecommendationRun

    def get_by_trace(self, trace_id: str) -> Optional[RecommendationRun]:
        return self.get_by(trace_id=trace_id)

    def list_by_product_wh(
        self, product_id: int, warehouse_id: int, *, limit: int = 50
    ) -> Sequence[RecommendationRun]:
        stmt = (
            select(RecommendationRun)
            .where(
                RecommendationRun.product_id == product_id,
                RecommendationRun.warehouse_id == warehouse_id,
            )
            .order_by(RecommendationRun.created_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def list_by_status(
        self, status: str, *, limit: int = 100
    ) -> Sequence[RecommendationRun]:
        return self.list(
            filters={"status": status},
            order_by=RecommendationRun.created_at.desc(),
            limit=limit,
        )

    def list_items(self, run_id: int) -> Sequence[RecommendationItem]:
        stmt = (
            select(RecommendationItem)
            .where(RecommendationItem.recommendation_run_id == run_id)
            .order_by(RecommendationItem.id.asc())
        )
        return list(self.session.execute(stmt).scalars().all())