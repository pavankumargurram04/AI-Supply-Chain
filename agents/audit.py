"""Persist agent messages to the audit log.

One row per message. The `entity_type` is `agent_message`, `entity_id` is
the recommendation_run id when available, and the payload is written to
`after`. This makes the full orchestration visible in the Audit Trail page.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from agents.messages import AgentMessage
from core.logging import get_logger
from db.models.audit_log import AuditLog

log = get_logger(__name__)


def persist_agent_message(
    session: Session,
    msg: AgentMessage,
    *,
    actor_id: Optional[int] = None,
    actor_name: Optional[str] = None,
    entity_id: Optional[int] = None,
) -> int:
    """Insert one AuditLog row for the message. Returns the row id."""
    row = AuditLog(
        actor_id=actor_id,
        actor_name=actor_name,
        action=f"agent.{msg.agent}",
        entity_type="agent_message",
        entity_id=entity_id,
        before=None,
        after=msg.to_dict(),
        request_id=None,
        trace_id=msg.trace_id,
        ip_address=None,
        user_agent=None,
        notes=None,
    )
    session.add(row)
    session.flush()
    return int(row.id)