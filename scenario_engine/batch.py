"""Batch scenario runner.

For a set of (product, warehouse) pairs, runs every scenario in a list and
persists the results. Deterministic given seed.

Usage:
    from scenario_engine import run_batch, SCENARIOS
    run_batch(
        scenario_keys=["demand_spike_50", "lead_time_plus_7", "inventory_500"],
        series=[("SKU-000000", "WH-NY")],
        mode="dev",
    )
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from core.database import session_scope
from core.logging import get_logger
from decision_engine.engine import DecisionEngine
from scenario_engine.catalog import SCENARIOS, get_scenario
from scenario_engine.runner import ScenarioRunResult, run_scenario

log = get_logger(__name__)


def run_batch(
    *,
    scenario_keys: Sequence[str],
    series: Sequence[Tuple[str, str]],
    mode: str = "dev",
    seed: int = 42,
    output_dir: str = "reports/scenarios",
    persist: bool = True,
) -> List[ScenarioRunResult]:
    """Run every (scenario, series) pair. Returns the list of results."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Build one engine and reuse it across scenarios (checkpoints load once).
    from apps.api.services import build_decision_engine
    engine: DecisionEngine = build_decision_engine(mode)

    results: List[ScenarioRunResult] = []
    specs = [get_scenario(k) for k in scenario_keys]

    if persist:
        with session_scope() as session:
            for sku, wh in series:
                for spec in specs:
                    r = run_scenario(
                        spec, product_sku=sku, warehouse_code=wh,
                        mode=mode, seed=seed, engine=engine,
                        persist_session=session,
                    )
                    results.append(r)
    else:
        for sku, wh in series:
            for spec in specs:
                r = run_scenario(
                    spec, product_sku=sku, warehouse_code=wh,
                    mode=mode, seed=seed, engine=engine,
                )
                results.append(r)

    # Write JSON + CSV summaries.
    (output / f"scenarios_{mode}.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2, default=str),
        encoding="utf-8",
    )

    csv_path = output / f"scenarios_{mode}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if results:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "scenario_key", "scenario_type", "product_sku", "warehouse_code",
                    "base_qty", "scen_qty", "delta_qty",
                    "base_cost", "scen_cost", "delta_cost",
                    "base_stockout", "scen_stockout", "delta_stockout",
                    "base_service", "scen_service", "delta_service",
                ],
            )
            w.writeheader()
            for r in results:
                w.writerow({
                    "scenario_key": r.scenario_key,
                    "scenario_type": r.scenario_type,
                    "product_sku": r.product_sku,
                    "warehouse_code": r.warehouse_code,
                    "base_qty": r.baseline["final_quantity"],
                    "scen_qty": r.scenario["final_quantity"],
                    "delta_qty": r.delta["final_quantity"],
                    "base_cost": r.baseline["total_cost"],
                    "scen_cost": r.scenario["total_cost"],
                    "delta_cost": r.delta["total_cost"],
                    "base_stockout": r.baseline["stockout_probability"],
                    "scen_stockout": r.scenario["stockout_probability"],
                    "delta_stockout": r.delta["stockout_probability"],
                    "base_service": r.baseline["expected_service_level"],
                    "scen_service": r.scenario["expected_service_level"],
                    "delta_service": r.delta["expected_service_level"],
                })

    log.info("scenario.batch.done", extra={
        "n_runs": len(results), "output": str(output),
    })
    return results