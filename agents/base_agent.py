"""Base agent class.

Every concrete agent:
    - receives a shared `context` dict (state, config, references)
    - calls a *specific* engine or model function
    - packages the result into an `AgentMessage`

Agents are stateless apart from the shared memory reference passed to their
constructor. They never invent numbers; if their dependency fails, they emit
a message with `status="error"` and let the coordinator decide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from agents.memory import AgentMemory
from agents.messages import AgentMessage, MessageStatus


class BaseAgent(ABC):
    name: str = "base"

    def __init__(self, memory: AgentMemory) -> None:
        self.memory = memory

    # -- Lifecycle ------------------------------------------------------
    def __call__(self, context: Dict[str, Any]) -> AgentMessage:
        trace_id = context.get("trace_id")
        parent_id = context.get("parent_id")
        if trace_id is None:
            raise ValueError(f"{self.name}: context missing 'trace_id'.")

        try:
            payload, model_versions, warnings = self.run(context)
            msg = AgentMessage(
                agent=self.name,
                trace_id=trace_id,
                parent_id=parent_id,
                payload=payload,
                model_versions=model_versions,
                warnings=warnings,
                status=MessageStatus.OK.value if not warnings else MessageStatus.WARNING.value,
            )
        except Exception as e:  # noqa: BLE001 - agents must not crash the pipeline silently
            msg = AgentMessage(
                agent=self.name,
                trace_id=trace_id,
                parent_id=parent_id,
                payload={},
                status=MessageStatus.ERROR.value,
                errors=[str(e)],
            )
        self.memory.record(msg)
        return msg

    # -- To be implemented by subclasses --------------------------------
    @abstractmethod
    def run(
        self, context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, str], list[str]]:
        """Return (payload, model_versions, warnings)."""
        raise NotImplementedError