"""Celery training tasks.

One entry point (`dispatch_training_job`) receives a training_jobs.id and
runs the appropriate trainer, updating the job row on completion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.database import session_scope
from core.errors import ValidationError
from core.logging import get_logger
from db.models.training_job import TrainingJob
from apps.worker.celery_app import celery_app

log = get_logger(__name__)


def _mark(job_id: int, **fields: Any) -> None:
    with session_scope() as s:
        job = s.get(TrainingJob, job_id)
        if job is None:
            return
        for k, v in fields.items():
            setattr(job, k, v)


@celery_app.task(bind=True, name="tasks.train.dispatch")
def dispatch_training_job(self, job_id: int) -> Dict[str, Any]:
    """Run the training job referenced by `job_id`."""
    with session_scope() as s:
        job = s.get(TrainingJob, job_id)
        if job is None:
            return {"ok": False, "error": f"job {job_id} not found"}
        family = job.model_family
        seed = job.seed or 42
        config = job.config or {}
        job.status = "running"
        job.started_at = datetime.now(tz=timezone.utc)
        s.flush()

    try:
        if family == "demand_lstm":
            from ml.demand.train import TrainConfig, train_demand_model
            mode = config.get("mode", "dev")
            cfg = TrainConfig(
                features_dir=f"datasets/curated/{mode}/features",
                output_dir=f"artifacts/demand/{mode}",
                seed=seed,
                max_series=config.get("max_series"),
            )
            meta = train_demand_model(cfg)
            metrics = {"val": meta["val_metrics"], "test": meta["test_metrics"]}

        elif family == "supplier_xgb":
            from ml.supplier_risk.train import SupplierTrainConfig, train_supplier_model
            mode = config.get("mode", "dev")
            cfg = SupplierTrainConfig(
                features_dir=f"datasets/curated/{mode}/features",
                raw_dir=f"datasets/raw/{mode}",
                output_dir=f"artifacts/supplier/{mode}",
                seed=seed,
            )
            meta = train_supplier_model(cfg)
            metrics = meta["metrics"]

        elif family == "inventory_dqn":
            from ml.inventory.train import DQNTrainConfig, train_inventory_dqn
            mode = config.get("mode", "dev")
            cfg = DQNTrainConfig(
                features_dir=f"datasets/curated/{mode}/features",
                output_dir=f"artifacts/inventory/{mode}",
                seed=seed,
            )
            meta = train_inventory_dqn(cfg)
            metrics = {"best_eval_cost": meta["best_eval_cost"],
                       "final_eval_cost": meta["final_eval_cost"]}
        else:
            raise ValidationError(f"Unknown model_family: {family}")

        _mark(
            job_id, status="succeeded",
            metrics=metrics,
            finished_at=datetime.now(tz=timezone.utc),
        )
        return {"ok": True, "metrics": metrics}

    except Exception as e:  # noqa: BLE001
        log.exception("training_job.failed", extra={"job_id": job_id})
        _mark(
            job_id, status="failed",
            error_message=str(e),
            finished_at=datetime.now(tz=timezone.utc),
        )
        return {"ok": False, "error": str(e)}