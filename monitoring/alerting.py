"""Alert payload + persistence.

A `MonitorReport` is a wrapper around one monitoring pass. Each monitor
returns a report; `write_alerts` persists them to the `alerts` table.

Alerts are deduplicated within a single `write_alerts` call — the same
(source, entity, kind) can only be emitted once. The DB layer does not
enforce uniqueness (historical alerts matter), so we dedupe in-process.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from core.enums import AlertSeverity, AlertStatus
from core.logging import get_logger
from db.models.alert import Alert

log = get_logger(__name__)


@dataclass
class AlertPayload:
    source: str                    # e.g. "forecast_drift"
    severity: str                  # info | warning | critical
    title: str
    message: str
    entity_type: Optional[str] = None
    entity_id: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def dedupe_key(self) -> str:
        raw = f"{self.source}|{self.entity_type}|{self.entity_id}|{self.title}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class MonitorReport:
    name: str
    alerts: List[AlertPayload] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metrics": dict(self.metrics),
            "notes": list(self.notes),
            "alerts": [a.to_dict() for a in self.alerts],
        }

    def add_alert(
        self,
        *,
        source: str,
        severity: str,
        title: str,
        message: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if severity not in (AlertSeverity.INFO.value, AlertSeverity.WARNING.value,
                            AlertSeverity.CRITICAL.value):
            raise ValueError(f"Invalid severity: {severity}")
        self.alerts.append(AlertPayload(
            source=source, severity=severity, title=title,
            message=message, entity_type=entity_type,
            entity_id=entity_id, payload=payload or {},
        ))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def write_alert(session: Session, alert: AlertPayload) -> int:
    row = Alert(
        severity=alert.severity,
        status=AlertStatus.OPEN.value,
        source=alert.source,
        title=alert.title,
        message=alert.message,
        entity_type=alert.entity_type,
        entity_id=alert.entity_id,
        payload=dict(alert.payload or {}),
    )
    session.add(row)
    session.flush()
    return int(row.id)


def write_alerts(session: Session, report: MonitorReport) -> List[int]:
    """Persist alerts from a report. Returns the DB ids (deduplicated within the report)."""
    seen: set[str] = set()
    ids: List[int] = []
    for a in report.alerts:
        key = a.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        ids.append(write_alert(session, a))
    log.info("monitoring.alerts.written", extra={"report": report.name, "n": len(ids)})
    return ids