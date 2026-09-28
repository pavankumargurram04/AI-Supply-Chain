"""Celery monitoring tasks.

One scheduled task that runs the three domain monitors and persists alerts.
"""

from __future__ import annotations

from typing import Any, Dict

from core.database import session_scope
from core.logging import get_logger
from monitoring.alerting import write_alerts
from monitoring.forecast_monitor import monitor_demand
from monitoring.inventory_monitor import monitor_inventory
from monitoring.supplier_monitor import monitor_supplier
from apps.worker.celery_app import celery_app

log = get_logger(__name__)


@celery_app.task(name="tasks.monitor.run")
def run_monitoring(mode: str = "dev") -> Dict[str, Any]:
    reports = [
        monitor_demand(features_dir=f"datasets/curated/{mode}/features"),
        monitor_supplier(features_dir=f"datasets/curated/{mode}/features"),
        monitor_inventory(),
    ]
    total = 0
    with session_scope() as s:
        for r in reports:
            ids = write_alerts(s, r)
            total += len(ids)
    return {"ok": True, "alerts_written": total,
            "reports": [r.to_dict() for r in reports]}