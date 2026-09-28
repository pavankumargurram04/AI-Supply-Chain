"""The decision-trace dataclass and request schema.

This is the payload persisted to `recommendation_runs` and served to the UI.
It contains *everything* needed to reproduce a decision byte-for-byte:
input snapshot, model versions, data/feature versions, seed, scenario id.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class DecisionRequest:
    product_sku: str
    warehouse_code: str
    service_level: float = 0.95
    horizon_days: int = 30
    lead_time_override_days: Optional[float] = None       # scenario input
    lead_time_std_override: Optional[float] = None        # scenario input
    demand_multiplier: float = 1.0                        # scenario input
    inventory_override: Optional[float] = None            # scenario input
    budget: Optional[float] = None
    max_suppliers: Optional[int] = None
    preferred_supplier_codes: Optional[List[str]] = None
    seed: int = 42
    scenario_id: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecisionTrace:
    # Identity
    trace_id: str
    run_at: str
    request: Dict[str, Any]

    # Provenance — every artifact referenced by the run.
    data_version: Optional[str] = None
    feature_version: Optional[str] = None
    demand_model_version: Optional[str] = None
    supplier_model_version: Optional[str] = None
    inventory_model_version: Optional[str] = None
    optimization_version: str = "procurement_alloc_v1"

    # Inputs snapshot (data rows actually used).
    input_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Demand
    forecast_point: List[float] = field(default_factory=list)
    forecast_std: List[float] = field(default_factory=list)
    forecast_dates: List[str] = field(default_factory=list)
    forecast_origin: Optional[str] = None

    # Supplier risk per candidate
    supplier_risks: List[Dict[str, Any]] = field(default_factory=list)

    # Inventory simulation + safety stock + ROP
    inventory_state: Dict[str, Any] = field(default_factory=dict)
    ltd_simulation: Dict[str, Any] = field(default_factory=dict)
    safety_stock: float = 0.0
    reorder_point: float = 0.0
    stockout_probability: float = 0.0
    expected_shortage: float = 0.0

    # DQN policy output
    dqn_action: Optional[Dict[str, Any]] = None

    # Procurement optimization
    candidate_offers: List[Dict[str, Any]] = field(default_factory=list)
    allocation: Optional[Dict[str, Any]] = None

    # Final recommendation
    final_quantity: float = 0.0
    final_suppliers: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    expected_service_level: float = 0.0
    confidence: float = 0.0

    # Human-readable, numeric-traceable explanation
    explanation: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Guardrails and warnings
    guardrail_violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)

    @classmethod
    def new(cls, request: DecisionRequest, trace_id: str) -> "DecisionTrace":
        return cls(
            trace_id=trace_id,
            run_at=datetime.now(tz=timezone.utc).isoformat(),
            request=request.to_dict(),
        )