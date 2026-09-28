"""Celery data pipeline tasks: generate, ingest, build features."""

from __future__ import annotations

from typing import Any, Dict

from core.logging import get_logger
from apps.worker.celery_app import celery_app

log = get_logger(__name__)


@celery_app.task(name="tasks.data.generate")
def generate_synthetic(mode: str, seed: int = 42) -> Dict[str, Any]:
    from scripts.generate_synthetic_data import generate_all
    from data_platform.generator.config import MODE_PRESETS
    cfg = MODE_PRESETS[mode](seed=seed)
    counts = generate_all(cfg)
    return {"ok": True, "mode": mode, "row_counts": counts}


@celery_app.task(name="tasks.data.ingest")
def ingest(mode: str) -> Dict[str, Any]:
    from scripts.ingest_data import ingest_all
    counts = ingest_all(f"datasets/raw/{mode}")
    return {"ok": True, "mode": mode, "inserted": counts}


@celery_app.task(name="tasks.data.build_features")
def build_features(mode: str) -> Dict[str, Any]:
    from data_platform.features.feature_store import build_all_features
    result = build_all_features(
        input_dir=f"datasets/raw/{mode}",
        output_dir=f"datasets/curated/{mode}",
    )
    return {"ok": True, "mode": mode,
            "feature_version": result.feature_version,
            "row_counts": result.row_counts}