"""Atomic rollback to a previous production model.

Given a model_name, we look up the current production row and the most
recently archived row. In one transaction:
    - the current production row is archived,
    - the archived row is promoted back to production.

The rollback is recorded in both rows' `extras.rollback_history` so the
audit trail survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from core.enums import ModelStatus
from core.logging import get_logger
from model_registry.registry import Registry, RegistryError

log = get_logger(__name__)


@dataclass
class RollbackResult:
    ok: bool
    model_name: str
    from_version: Optional[str]
    to_version: Optional[str]
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "model_name": self.model_name,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "reason": self.reason,
        }


def rollback(
    registry: Registry,
    *,
    model_name: str,
    to_version: Optional[str] = None,
    actor: Optional[str] = None,
    reason: str = "",
) -> RollbackResult:
    """Rollback the model.

    If `to_version` is None we roll back to the most recently archived version
    of the same model.
    """
    current = registry.get_production(model_name)
    if current is None:
        return RollbackResult(
            ok=False, model_name=model_name, from_version=None,
            to_version=to_version, reason="no current production model",
        )

    if to_version is not None:
        target = registry.get(model_name=model_name, version=to_version)
        if target is None:
            return RollbackResult(
                ok=False, model_name=model_name,
                from_version=current.version, to_version=to_version,
                reason=f"target version {to_version} not found",
            )
    else:
        target = _latest_archived(registry, model_name)
        if target is None:
            return RollbackResult(
                ok=False, model_name=model_name,
                from_version=current.version, to_version=None,
                reason="no archived version available to roll back to",
            )
        to_version = target.version

    if target.id == current.id:
        return RollbackResult(
            ok=False, model_name=model_name,
            from_version=current.version, to_version=to_version,
            reason="target is the current production version",
        )

    now = datetime.now(tz=timezone.utc)

    # Archive current.
    current.status = ModelStatus.ARCHIVED.value
    current.archived_at = now
    cur_extras = dict(current.extras or {})
    cur_extras.setdefault("rollback_history", []).append({
        "at": now.isoformat(), "actor": actor,
        "action": "archived_due_to_rollback",
        "target_version": to_version, "reason": reason,
    })
    current.extras = cur_extras

    # Promote target.
    target.status = ModelStatus.PRODUCTION.value
    target.promoted_at = now
    tgt_extras = dict(target.extras or {})
    tgt_extras.setdefault("rollback_history", []).append({
        "at": now.isoformat(), "actor": actor,
        "action": "promoted_due_to_rollback",
        "from_version": current.version, "reason": reason,
    })
    target.extras = tgt_extras

    registry.session.flush()
    log.info("registry.rollback.ok", extra={
        "model_name": model_name,
        "from": current.version, "to": target.version,
    })
    return RollbackResult(
        ok=True, model_name=model_name,
        from_version=current.version, to_version=target.version,
        reason=reason,
    )


def _latest_archived(registry: Registry, model_name: str):
    from db.models.model_registry import ModelRegistry
    return (
        registry.session.query(ModelRegistry)
        .filter_by(model_name=model_name, status=ModelStatus.ARCHIVED.value)
        .order_by(ModelRegistry.archived_at.desc().nullslast())
        .first()
    )