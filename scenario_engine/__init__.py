"""Scenario engine: predefined + custom scenarios executed through the
decision engine with full provenance.

Public API:
    from scenario_engine import SCENARIOS, ScenarioSpec, run_scenario, run_batch

    spec = SCENARIOS["demand_spike_50"]
    result = run_scenario(spec, product_sku="SKU-000000", warehouse_code="WH-NY", mode="dev")
"""

from scenario_engine.catalog import SCENARIOS, ScenarioSpec, get_scenario
from scenario_engine.runner import ScenarioRunResult, run_scenario
from scenario_engine.batch import run_batch

__all__ = [
    "SCENARIOS",
    "ScenarioSpec",
    "get_scenario",
    "ScenarioRunResult",
    "run_scenario",
    "run_batch",
]