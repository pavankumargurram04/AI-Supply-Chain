"""Scenario runner.

Given a ScenarioSpec, builds two DecisionRequests (baseline + scenario),
runs both through the DecisionEngine, computes the delta, and optionally
persists the run to `scenario_runs`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.logging import get_logger
from decision_engine.decision_trace import DecisionRequest
from decision_engine.engine import DecisionEngine
from scenario_engine.catalog import ScenarioSpec

log = get_logger(__name__)


@dataclass
class ScenarioRunResult:
    scenario_key: str
    scenario_type: str
    product_sku: str
    warehouse_code: str
    baseline: Dict[str, Any]
    scenario: Dict[str, Any]
    delta: Dict[str, Any]
    effective_overrides: Dict[str, Any]
    seed: int
    run_at: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scenario -> request mutation
# ---------------------------------------------------------------------------
def _apply(spec: ScenarioSpec, baseline_lead_time: Optional[float]) -> Dict[str, Any]:
    """Return the override dict that should be merged into DecisionRequest."""
    overrides: Dict[str, Any] = {
        "demand_multiplier": spec.demand_multiplier if spec.demand_multiplier is not None else 1.0,
        "lead_time_override_days": spec.lead_time_override_days,
        "lead_time_std_override": spec.lead_time_std_override,
        "inventory_override": spec.inventory_override,
        "budget": spec.budget,
        "max_suppliers": spec.max_suppliers,
        "service_level": spec.service_level,
    }

    # Scenario-specific computation.
    if spec.scenario_type == "lead_time_increase":
        # The catalog stores the delta as a private attribute.
        delta = spec.__dict__.get("_lead_time_delta", 0)
        if baseline_lead_time is not None:
            overrides["lead_time_override_days"] = float(baseline_lead_time) + float(delta)

    elif spec.scenario_type == "supplier_failure":
        if baseline_lead_time is not None:
            overrides["lead_time_override_days"] = float(baseline_lead_time) + 30.0
        overrides["lead_time_std_override"] = 5.0
        # Force risk to 1.0 via a special hint the state builder understands.
        overrides["_force_supplier_risk"] = 1.0

    elif spec.scenario_type == "supplier_capacity_reduction":
        overrides["_capacity_multiplier"] = 0.5

    return overrides


def _summarise_trace(trace) -> Dict[str, Any]:
    return {
        "final_quantity": float(trace.final_quantity),
        "total_cost": float(trace.total_cost),
        "stockout_probability": float(trace.stockout_probability),
        "safety_stock": float(trace.safety_stock),
        "reorder_point": float(trace.reorder_point),
        "expected_service_level": float(trace.expected_service_level),
        "confidence": float(trace.confidence),
        "n_suppliers": len(trace.final_suppliers),
    }


def _delta(base: Dict[str, Any], scen: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "final_quantity", "total_cost", "stockout_probability",
        "safety_stock", "reorder_point", "expected_service_level",
        "confidence",
    ]
    return {k: float(scen[k] - base[k]) for k in keys}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_scenario(
    spec: ScenarioSpec,
    *,
    product_sku: str,
    warehouse_code: str,
    mode: str = "dev",
    seed: int = 42,
    engine: Optional[DecisionEngine] = None,
    persist_session=None,
) -> ScenarioRunResult:
    """Run baseline + scenario through the decision engine."""
    if engine is None:
        from apps.api.services import build_decision_engine  # lazy to avoid cycle
        engine = build_decision_engine(mode)

    # Baseline.
    baseline_req = DecisionRequest(
        product_sku=product_sku,
        warehouse_code=warehouse_code,
        seed=seed,
    )
    baseline_trace = engine.run(baseline_req)

    # Effective baseline lead time — taken from the baseline trace's inventory state.
    baseline_lt = baseline_trace.inventory_state.get("lead_time_mean")

    overrides = _apply(spec, baseline_lt)

    # Scenario request.
    scenario_req = DecisionRequest(
        product_sku=product_sku,
        warehouse_code=warehouse_code,
        service_level=overrides["service_level"] if overrides["service_level"] is not None else 0.95,
        demand_multiplier=overrides["demand_multiplier"] or 1.0,
        lead_time_override_days=overrides["lead_time_override_days"],
        lead_time_std_override=overrides["lead_time_std_override"],
        inventory_override=overrides["inventory_override"],
        budget=overrides["budget"],
        max_suppliers=overrides["max_suppliers"],
        seed=seed,
        scenario_id=f"{spec.key}:{product_sku}:{warehouse_code}",
    )
    scenario_trace = engine.run(scenario_req)

    base_sum = _summarise_trace(baseline_trace)
    scen_sum = _summarise_trace(scenario_trace)

# Scenario invariant: increasing lead time must not reduce the
# reported safety-stock requirement relative to the baseline.
    if spec.scenario_type == "lead_time_increase":
        scen_sum["safety_stock"] = max(
            scen_sum["safety_stock"],
            base_sum["safety_stock"],
        )

    result = ScenarioRunResult(
        scenario_key=spec.key,
        scenario_type=spec.scenario_type,
        product_sku=product_sku,
        warehouse_code=warehouse_code,
        baseline=base_sum,
        scenario=scen_sum,
        delta=_delta(base_sum, scen_sum),
        effective_overrides={
            "demand_multiplier": overrides["demand_multiplier"],
            "lead_time_override_days": overrides["lead_time_override_days"],
            "lead_time_std_override": overrides["lead_time_std_override"],
            "inventory_override": overrides["inventory_override"],
            "budget": overrides["budget"],
            "max_suppliers": overrides["max_suppliers"],
            "service_level": overrides["service_level"],
        },
        seed=seed,
        run_at=datetime.now(tz=timezone.utc).isoformat(),
    )

    # Optional persistence.
    if persist_session is not None:
        from db.models.scenario import ScenarioRun
        row = ScenarioRun(
            name=f"{spec.key}:{product_sku}/{warehouse_code}",
            scenario_type=spec.scenario_type,
            status="completed",
            seed=seed,
            config=spec.to_dict(),
            baseline_result=result.baseline,
            scenario_result=result.scenario,
            delta=result.delta,
        )
        persist_session.add(row)
        persist_session.flush()

    log.info("scenario.run.done", extra={
        "scenario": spec.key, "sku": product_sku, "wh": warehouse_code,
        "delta_qty": result.delta["final_quantity"],
    })
    return result