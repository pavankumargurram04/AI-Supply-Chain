"""Pydantic schemas for API integration.

These mirror the dataclass messages and are used by the FastAPI layer to
serialise/deserialise agent traffic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentMessageSchema(BaseModel):
    message_id: str
    trace_id: str = ""
    parent_id: Optional[str] = None
    agent: str
    timestamp: str
    model_versions: Dict[str, str] = Field(default_factory=dict)
    input_refs: List[Dict[str, Any]] = Field(default_factory=list)
    payload: Dict[str, Any]
    confidence: Optional[float] = None
    status: str = "ok"
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class OrchestratorRequestSchema(BaseModel):
    product_sku: str
    warehouse_code: str
    service_level: float = 0.95
    horizon_days: int = 30
    demand_multiplier: float = 1.0
    lead_time_override_days: Optional[float] = None
    lead_time_std_override: Optional[float] = None
    inventory_override: Optional[float] = None
    budget: Optional[float] = None
    max_suppliers: Optional[int] = None
    preferred_supplier_codes: Optional[List[str]] = None
    seed: int = 42


class OrchestratorResultSchema(BaseModel):
    trace_id: str
    run_at: str
    messages: List[AgentMessageSchema]
    final_quantity: float
    total_cost: float
    expected_service_level: float
    stockout_probability: float
    explanation: str
    confidence: float
    guardrail_violations: List[str]