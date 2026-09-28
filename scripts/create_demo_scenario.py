"""CLI: run a full decision-engine cycle for one (product, warehouse).

Usage:
    python scripts\create_demo_scenario.py --mode dev --product SKU-000000 --warehouse WH-NY
    python scripts\create_demo_scenario.py --mode dev --product SKU-000000 --warehouse WH-NY --demand-multiplier 1.5 --lead-time-override 14

The script prints the recommendation, saves the trace to a JSON file, and
optionally persists it to the DB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logging import configure_logging  # noqa: E402
from decision_engine.decision_trace import DecisionRequest  # noqa: E402
from decision_engine.engine import DecisionEngine  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Run a decision-engine cycle for one series.")
    p.add_argument("--mode", choices=["dev", "research", "stress"], default="dev")
    p.add_argument("--product", required=True)
    p.add_argument("--warehouse", required=True)
    p.add_argument("--service-level", type=float, default=0.95)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--demand-multiplier", type=float, default=1.0)
    p.add_argument("--lead-time-override", type=float, default=None)
    p.add_argument("--lead-time-std-override", type=float, default=None)
    p.add_argument("--inventory-override", type=float, default=None)
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--max-suppliers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario-id", type=str, default=None)
    p.add_argument("--persist", action="store_true",
                   help="Also write the trace to the DB (recommendation_runs).")
    args = p.parse_args()

    configure_logging()

    mode = args.mode
    engine = DecisionEngine(
        demand_checkpoint_dir=f"artifacts/demand/{mode}",
        supplier_checkpoint_dir=f"artifacts/supplier/{mode}",
        inventory_checkpoint_dir=f"artifacts/inventory/{mode}",
        curated_features_dir=f"datasets/curated/{mode}/features",
        raw_dir=f"datasets/raw/{mode}",
        simulation_samples=10_000,
        device="cpu",
    )

    request = DecisionRequest(
        product_sku=args.product,
        warehouse_code=args.warehouse,
        service_level=args.service_level,
        horizon_days=args.horizon,
        demand_multiplier=args.demand_multiplier,
        lead_time_override_days=args.lead_time_override,
        lead_time_std_override=args.lead_time_std_override,
        inventory_override=args.inventory_override,
        budget=args.budget,
        max_suppliers=args.max_suppliers,
        seed=args.seed,
        scenario_id=args.scenario_id,
    )

    trace = engine.run(request)

    # Save the trace.
    out_dir = Path("reports/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"decision_trace_{trace.trace_id}.json"
    out_path.write_text(trace.to_json(), encoding="utf-8")

    # Print a summary.
    summary = {
        "trace_id": trace.trace_id,
        "product_sku": request.product_sku,
        "warehouse_code": request.warehouse_code,
        "final_quantity": trace.final_quantity,
        "total_cost": trace.total_cost,
        "expected_service_level": trace.expected_service_level,
        "stockout_probability": trace.stockout_probability,
        "safety_stock": trace.safety_stock,
        "reorder_point": trace.reorder_point,
        "n_suppliers": len(trace.final_suppliers),
        "confidence": trace.confidence,
        "guardrail_violations": trace.guardrail_violations,
        "trace_path": str(out_path),
    }
    print(json.dumps(summary, indent=2, default=str))

    # Print the explanation verbatim.
    print("\n--- Explanation ---")
    print(trace.explanation)

    # Optionally persist.
    if args.persist:
        from core.database import session_scope
        with session_scope() as session:
            rid = engine.persist(trace, session)
            print(f"\nPersisted to recommendation_runs.id = {rid}")

    return 0 if not trace.guardrail_violations else 2


if __name__ == "__main__":
    raise SystemExit(main())    