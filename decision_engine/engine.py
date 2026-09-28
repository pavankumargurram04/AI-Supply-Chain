"""Top-level DecisionEngine.

Wires all the individual modules into one call. Persists a DecisionTrace
row in `recommendation_runs` when a DB session is supplied.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.logging import get_logger
from decision_engine.confidence_engine import compute_confidence
from decision_engine.decision_trace import DecisionRequest, DecisionTrace
from decision_engine.explanation_engine import build_explanation
from decision_engine import forecast_decision
from decision_engine.inventory_decision import run_inventory_decision
from decision_engine.policy_guardrails import check_guardrails
from decision_engine.procurement_optimizer import optimize
from decision_engine.state_builder import build_state
from decision_engine import supplier_decision

log = get_logger(__name__)


class DecisionEngine:
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

    # -- Public API -----------------------------------------------------
    def run(self, request: DecisionRequest) -> DecisionTrace:
        trace_id = request.scenario_id or str(uuid.uuid4())
        trace = DecisionTrace.new(request, trace_id=request.scenario_id or trace_id)

        # 1) Build state.
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
        trace.input_snapshot = state.snapshot()
        trace.data_version = state.data_version
        trace.feature_version = state.feature_version

        # 2) Forecast.
        forecast = forecast_decision.produce_forecast(
            state, demand_checkpoint_dir=self.demand_checkpoint_dir, device=self.device,
        )
        trace.forecast_point = [float(x) for x in forecast.point]
        trace.forecast_std = [float(x) for x in forecast.std]
        trace.forecast_dates = list(forecast.dates)
        trace.forecast_origin = forecast.origin_date
        trace.demand_model_version = forecast.model_version

        # 3) Supplier risk.
        supplier_risks = supplier_decision.score_suppliers(
            state, supplier_checkpoint_dir=self.supplier_checkpoint_dir,
        )
        trace.supplier_risks = supplier_risks
        if supplier_risks:
            trace.supplier_model_version = supplier_risks[0].get("model_version")

        # 4) Inventory decision.
        inventory = run_inventory_decision(
            state, forecast,
            simulation_samples=self.simulation_samples,
        )
        trace.inventory_state = {
            "available": inventory.available,
            "inventory_position": inventory.inventory_position,
            "expected_lead_time_demand": inventory.expected_lead_time_demand,
            "lead_time_mean": inventory.lead_time_mean,
            "lead_time_std": inventory.lead_time_std,
        }
        trace.ltd_simulation = inventory.ltd_simulation
        trace.safety_stock = inventory.safety_stock
        trace.reorder_point = inventory.reorder_point
        trace.stockout_probability = inventory.stockout_probability
        trace.expected_shortage = inventory.expected_shortage

        # 5) Procurement optimization.
        opt = optimize(
            state,
            inventory,
            budget=request.budget,
            max_suppliers=request.max_suppliers,
            preferred_supplier_codes=request.preferred_supplier_codes,
            inventory_checkpoint_dir=self.inventory_checkpoint_dir,
        )
        trace.candidate_offers = []
        for c in state.suppliers:
            trace.candidate_offers.append({
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
            })
        trace.allocation = opt["allocation"]
        trace.final_quantity = float(opt["final_quantity"])
        trace.final_suppliers = list(opt["allocation"].get("lines", []))
        trace.total_cost = float(opt["allocation"].get("total_cost", 0.0))
        trace.expected_service_level = float(opt["expected_service_level"])

        # DQN action (if the policy was used).
        for c in opt["candidates"]:
            if c["source"] == "dqn":
                trace.dqn_action = c["details"]
                trace.inventory_model_version = c["details"].get("model_version")

        # 6) Guardrails.
        guard = check_guardrails(
            recommended_quantity=trace.final_quantity,
            allocation_total_cost=trace.total_cost,
            budget=request.budget,
            service_level_target=request.service_level,
            predicted_service_level=trace.expected_service_level,
            lead_time_days=inventory.lead_time_mean,
        )
        trace.guardrail_violations = list(guard.violations)
        if not guard.passed:
            trace.warnings.append("guardrails violated — human review required")

        # 7) Confidence.
        trace.confidence = compute_confidence(trace.to_dict())

        # 8) Explanation.
        explanation = build_explanation(trace.to_dict())
        trace.explanation = explanation["explanation"]
        trace.evidence = explanation["evidence"]

        return trace

    # -- Persist --------------------------------------------------------
    def persist(self, trace: DecisionTrace, session) -> int:
        """Write the trace to the DB; returns the recommendation_runs.id."""
        from db.models.recommendation import RecommendationRun

        run = RecommendationRun(
            trace_id=trace.trace_id,
            product_id=0,   # resolved by caller if needed
            warehouse_id=0,
            status="generated",
            request=trace.request,
            input_snapshot=trace.input_snapshot,
            forecast={
                "point": trace.forecast_point,
                "std": trace.forecast_std,
                "dates": trace.forecast_dates,
                "origin_date": trace.forecast_origin,
            },
            supplier_risks={"rows": trace.supplier_risks},
            inventory_state=trace.inventory_state,
            dqn_action=trace.dqn_action,
            optimization_result=trace.allocation,
            decision_trace=trace.to_dict(),
            stockout_probability=trace.stockout_probability,
            safety_stock=trace.safety_stock,
            reorder_point=trace.reorder_point,
            final_quantity=trace.final_quantity,
            supplier_allocation={"lines": trace.final_suppliers},
            expected_cost=trace.total_cost,
            expected_service_level=trace.expected_service_level,
            explanation=trace.explanation,
            confidence=trace.confidence,
            demand_model_version=trace.demand_model_version,
            supplier_model_version=trace.supplier_model_version,
            inventory_model_version=trace.inventory_model_version,
            optimization_version=trace.optimization_version,
            data_version=trace.data_version,
            feature_version=trace.feature_version,
        )
        session.add(run)
        session.flush()
        return int(run.id)
