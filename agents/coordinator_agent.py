"""Coordinator agent.

Runs the other agents in dependency order, collects their messages, applies
guardrails, computes confidence, produces the explanation, and returns a
final `OrchestratorResult`.

Dependency graph:
    DemandAgent
        ↓
    InventoryAgent  ← SupplierAgent
        ↓
    ProcurementAgent
        ↓
    Coordinator (finalize)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.demand_agent import DemandAgent
from agents.inventory_agent import InventoryAgent
from agents.memory import AgentMemory
from agents.messages import AgentMessage
from agents.procurement_agent import ProcurementAgent
from agents.supplier_agent import SupplierAgent
from core.logging import get_logger
from decision_engine.confidence_engine import compute_confidence
from decision_engine.decision_trace import DecisionRequest, DecisionTrace
from decision_engine.explanation_engine import build_explanation
from decision_engine.policy_guardrails import check_guardrails
from decision_engine.state_builder import build_state

log = get_logger(__name__)


@dataclass
class OrchestratorResult:
    trace_id: str
    run_at: str
    messages: List[AgentMessage]
    decision_trace: DecisionTrace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "run_at": self.run_at,
            "messages": [m.to_dict() for m in self.messages],
            "decision_trace": self.decision_trace.to_dict(),
        }


class AgentOrchestrator:
    def __init__(
        self,
        *,
        demand_checkpoint_dir: str | Path,
        supplier_checkpoint_dir: str | Path,
        inventory_checkpoint_dir: Optional[str | Path],
        curated_features_dir: str | Path,
        raw_dir: Optional[str | Path] = None,
        simulation_samples: int = 10_000,
        device: str = "cpu",
    ) -> None:
        self.demand_checkpoint_dir = Path(demand_checkpoint_dir)
        self.supplier_checkpoint_dir = Path(supplier_checkpoint_dir)
        self.inventory_checkpoint_dir = Path(inventory_checkpoint_dir) if inventory_checkpoint_dir else None
        self.curated_features_dir = Path(curated_features_dir)
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.simulation_samples = simulation_samples
        self.device = device

    def run(self, request: DecisionRequest) -> OrchestratorResult:
        trace_id = request.scenario_id or uuid.uuid4().hex
        memory = AgentMemory()
        memory.start_trace(trace_id)

        # 1) Build the shared state.
        state = build_state(
            product_sku=request.product_sku,
            warehouse_code=request.warehouse_code,
            curated_features_dir=self.curated_features_dir,
            raw_dir=self.raw_dir,
            service_level=request.service_level,
            horizon_days=request.horizon_days,
            seed=request.seed,
            demand_multiplier=request.demand_multiplier,
            lead_time_override_days=request.lead_time_override_days,
            lead_time_std_override=request.lead_time_std_override,
            inventory_override=request.inventory_override,
        )

        # Shared context.
        ctx_base: Dict[str, Any] = {
            "trace_id": trace_id,
            "state": state,
            "request": request,
            "demand_checkpoint_dir": self.demand_checkpoint_dir,
            "supplier_checkpoint_dir": self.supplier_checkpoint_dir,
            "inventory_checkpoint_dir": self.inventory_checkpoint_dir,
            "simulation_samples": self.simulation_samples,
            "device": self.device,
        }

        # 2) Demand.
        demand_agent = DemandAgent(memory)
        demand_msg = demand_agent(ctx_base)
        ctx_base["demand_message"] = demand_msg
        ctx_base["parent_id"] = demand_msg.message_id

        # 3) Supplier risk.
        supplier_agent = SupplierAgent(memory)
        supplier_msg = supplier_agent(ctx_base)
        ctx_base["supplier_message"] = supplier_msg

        # 4) Inventory (needs demand).
        inventory_agent = InventoryAgent(memory)
        inventory_msg = inventory_agent(ctx_base)
        ctx_base["inventory_message"] = inventory_msg
        ctx_base["parent_id"] = inventory_msg.message_id

        # 5) Procurement (needs inventory + state).
        procurement_agent = ProcurementAgent(memory)
        procurement_msg = procurement_agent(ctx_base)

        # 6) Finalise.
        all_messages = list(memory.get(trace_id).messages) if memory.get(trace_id) else []

        decision_trace = self._finalise(
            request=request,
            state=state,
            trace_id=trace_id,
            messages=all_messages,
        )

        return OrchestratorResult(
            trace_id=trace_id,
            run_at=datetime.now(tz=timezone.utc).isoformat(),
            messages=all_messages,
            decision_trace=decision_trace,
        )

    # ------------------------------------------------------------------
    def _finalise(
        self,
        *,
        request: DecisionRequest,
        state,
        trace_id: str,
        messages: List[AgentMessage],
    ) -> DecisionTrace:
        """Merge agent messages into a single DecisionTrace."""
        trace = DecisionTrace.new(request, trace_id=trace_id)
        trace.input_snapshot = state.snapshot()
        trace.data_version = state.data_version
        trace.feature_version = state.feature_version

        by_agent: Dict[str, AgentMessage] = {}
        for m in messages:
            by_agent[m.agent] = m

        # Demand
        d = by_agent.get("demand")
        if d and d.status != "error":
            p = d.payload
            trace.forecast_point = list(p.get("point", []))
            trace.forecast_std = list(p.get("std", []))
            trace.forecast_dates = list(p.get("dates", []))
            trace.forecast_origin = p.get("origin_date")
            trace.demand_model_version = p.get("model_version")

        # Supplier
        s = by_agent.get("supplier")
        if s and s.status != "error":
            trace.supplier_risks = list(s.payload.get("suppliers", []))
            if trace.supplier_risks:
                trace.supplier_model_version = trace.supplier_risks[0].get("model_version")

        # Inventory
        i = by_agent.get("inventory")
        if i and i.status != "error":
            p = i.payload
            trace.inventory_state = {
                "available": p.get("available"),
                "inventory_position": p.get("inventory_position"),
                "expected_lead_time_demand": p.get("expected_lead_time_demand"),
                "lead_time_mean": p.get("lead_time_mean"),
                "lead_time_std": p.get("lead_time_std"),
            }
            trace.ltd_simulation = dict(p.get("ltd_simulation", {}))
            trace.safety_stock = float(p.get("safety_stock", 0.0))
            trace.reorder_point = float(p.get("reorder_point", 0.0))
            trace.stockout_probability = float(p.get("stockout_probability", 0.0))
            trace.expected_shortage = float(p.get("expected_shortage", 0.0))

        # Procurement
        p = by_agent.get("procurement")
        if p and p.status != "error":
            payload = p.payload
            trace.allocation = payload.get("allocation")
            trace.final_quantity = float(payload.get("final_quantity", 0.0))
            trace.final_suppliers = list((payload.get("allocation") or {}).get("lines", []))
            trace.total_cost = float((payload.get("allocation") or {}).get("total_cost", 0.0))
            trace.expected_service_level = float(payload.get("expected_service_level", 0.0))
            for c in payload.get("candidates", []):
                if c.get("source") == "dqn":
                    trace.dqn_action = c.get("details")
                    trace.inventory_model_version = (c.get("details") or {}).get("model_version")

        trace.candidate_offers = [
            {
                "supplier_code": c.supplier_code,
                "unit_price": float(c.unit_price),
                "moq": int(c.moq),
                "max_order_qty": int(c.max_order_qty),
                "shipping_cost": float(c.shipping_cost),
                "lead_time_days": float(c.lead_time_days),
                "lead_time_std_days": float(c.lead_time_std_days),
                "risk_probability": float(c.risk_probability),
                "risk_label": c.risk_label,
                "is_preferred": bool(c.is_preferred),
            }
            for c in state.suppliers
        ]

        # Guardrails.
        guard = check_guardrails(
            recommended_quantity=trace.final_quantity,
            allocation_total_cost=trace.total_cost,
            budget=request.budget,
            service_level_target=request.service_level,
            predicted_service_level=trace.expected_service_level,
            lead_time_days=float(trace.inventory_state.get("lead_time_mean", 1.0) or 1.0),
        )
        trace.guardrail_violations = list(guard.violations)
        for m in messages:
            trace.warnings.extend(m.warnings)

        # Confidence + explanation.
        trace.confidence = compute_confidence(trace.to_dict())
        explanation = build_explanation(trace.to_dict())
        trace.explanation = explanation["explanation"]
        trace.evidence = explanation["evidence"]

        return trace