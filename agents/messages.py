"""Structured agent messages.

Every message carries:
    - message_id      : unique per message (UUID4 hex)
    - trace_id        : shared across a single orchestration run
    - parent_id       : message_id of the message this was derived from (optional)
    - agent           : the agent that produced this message
    - timestamp       : ISO-8601 UTC
    - model_versions  : dict of {model_name: version_string}
    - input_refs      : references to inputs (paths, IDs, snapshot hashes)
    - payload         : the actual content — a JSON-serialisable dict
    - confidence      : float in [0, 1], optional
    - status          : "ok" | "warning" | "error"
    - warnings        : list of human-readable strings
    - errors          : list of human-readable strings

The `payload` schema is documented per agent. Payloads are *data only* — no
business logic. Logic lives in the agent classes.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    DEMAND = "demand"
    SUPPLIER = "supplier"
    INVENTORY = "inventory"
    PROCUREMENT = "procurement"


class MessageStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class AgentMessage:
    agent: str
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str = ""
    parent_id: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    model_versions: Dict[str, str] = field(default_factory=dict)
    input_refs: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    status: str = MessageStatus.OK.value
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)

    def summary(self) -> Dict[str, Any]:
        """Compact representation for logging / UI."""
        return {
            "message_id": self.message_id,
            "agent": self.agent,
            "status": self.status,
            "confidence": self.confidence,
            "n_payload_keys": len(self.payload),
            "n_warnings": len(self.warnings),
            "n_errors": len(self.errors),
            "timestamp": self.timestamp,
        }