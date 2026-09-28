"""In-process + DB-backed message store.

Two layers:
    - InMemoryStore: keeps messages for the current run (used by the UI).
    - DB persistence: optionally writes each message to the `audit_logs` table.

The memory is keyed by `trace_id`; a single orchestration run owns one trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.logging import get_logger
from agents.messages import AgentMessage

log = get_logger(__name__)


@dataclass
class TraceRecord:
    trace_id: str
    messages: List[AgentMessage] = field(default_factory=list)

    def add(self, msg: AgentMessage) -> None:
        self.messages.append(msg)

    def by_agent(self, agent: str) -> List[AgentMessage]:
        return [m for m in self.messages if m.agent == agent]

    def latest(self, agent: str) -> Optional[AgentMessage]:
        matching = self.by_agent(agent)
        return matching[-1] if matching else None

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "n_messages": len(self.messages),
            "messages": [m.to_dict() for m in self.messages],
        }


class AgentMemory:
    """In-process memory of agent messages keyed by trace_id."""

    def __init__(self) -> None:
        self._traces: Dict[str, TraceRecord] = {}

    def start_trace(self, trace_id: str) -> TraceRecord:
        record = TraceRecord(trace_id=trace_id)
        self._traces[trace_id] = record
        return record

    def get(self, trace_id: str) -> Optional[TraceRecord]:
        return self._traces.get(trace_id)

    def record(self, msg: AgentMessage) -> None:
        if not msg.trace_id:
            raise ValueError("AgentMessage.trace_id must be set before recording.")
        rec = self._traces.get(msg.trace_id)
        if rec is None:
            rec = self.start_trace(msg.trace_id)
        rec.add(msg)

    def clear(self, trace_id: Optional[str] = None) -> None:
        if trace_id is None:
            self._traces.clear()
        else:
            self._traces.pop(trace_id, None)

    def list_traces(self) -> List[str]:
        return list(self._traces.keys())