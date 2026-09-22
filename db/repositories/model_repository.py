"""Repository for ModelRegistry and TrainingJob."""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select

from core.enums import ModelStatus, TrainingJobStatus
from db.models.model_registry import ModelRegistry
from db.models.training_job import TrainingJob
from db.repositories.base_repository import BaseRepository


class ModelRepository(BaseRepository[ModelRegistry]):
    model = ModelRegistry

    def get_version(self, model_name: str, version: str) -> Optional[ModelRegistry]:
        return self.get_by(model_name=model_name, version=version)

    def list_production(self) -> Sequence[ModelRegistry]:
        return self.list(
            filters={"status": ModelStatus.PRODUCTION.value},
            order_by=ModelRegistry.promoted_at.desc(),
        )

    def latest_for_family(
        self, model_family: str, *, status: Optional[str] = None
    ) -> Optional[ModelRegistry]:
        stmt = select(ModelRegistry).where(ModelRegistry.model_family == model_family)
        if status is not None:
            stmt = stmt.where(ModelRegistry.status == status)
        stmt = stmt.order_by(ModelRegistry.created_at.desc()).limit(1)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_for_family(self, model_family: str) -> Sequence[ModelRegistry]:
        stmt = (
            select(ModelRegistry)
            .where(ModelRegistry.model_family == model_family)
            .order_by(ModelRegistry.created_at.desc())
        )
        return list(self.session.execute(stmt).scalars().all())

    # ---- Training jobs ----
    def create_training_job(self, **fields) -> TrainingJob:
        job = TrainingJob(**fields)
        self.session.add(job)
        self.session.flush()
        return job

    def get_training_job(self, job_id: int) -> Optional[TrainingJob]:
        return self.session.get(TrainingJob, job_id)

    def list_training_jobs(
        self, *, status: Optional[str] = None, limit: int = 100
    ) -> Sequence[TrainingJob]:
        stmt = select(TrainingJob)
        if status is not None:
            stmt = stmt.where(TrainingJob.status == status)
        stmt = stmt.order_by(TrainingJob.created_at.desc()).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def list_queued_jobs(self) -> Sequence[TrainingJob]:
        return self.list_training_jobs(status=TrainingJobStatus.QUEUED.value)