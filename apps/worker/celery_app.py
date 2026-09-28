"""Celery application factory.

Configuration is read from Settings. Task modules auto-register.
"""

from __future__ import annotations

from celery import Celery

from core.config import get_settings


def create_celery() -> Celery:
    settings = get_settings()
    app = Celery(
        "ai_supply_chain",
        broker=settings.celery_broker,
        backend=settings.celery_backend,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_routes={
            "tasks.train.*": {"queue": "training"},
            "tasks.forecast.*": {"queue": "forecasting"},
            "tasks.data.*": {"queue": "data"},
            "tasks.monitor.*": {"queue": "monitoring"},
        },
        task_default_queue="default",
    )
    return app


celery_app = create_celery()

import apps.worker.tasks.training_tasks
import apps.worker.tasks.forecasting_tasks
import apps.worker.tasks.data_tasks
import apps.worker.tasks.monitoring_tasks
