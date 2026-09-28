"""Seed N recommendations into the DB for demo purposes.

Bypasses HTTP/auth and writes directly to the database using the same
decision-engine code path as POST /api/v1/procurement/recommend.

Usage:
    python scripts\\seed_recommendations.py --mode dev --n 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.api.services import build_decision_engine  # noqa: E402
from core.database import session_scope  # noqa: E402
from core.enums import RecommendationStatus  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402
from db.models.audit_log import AuditLog  # noqa: E402
from db.models.model_registry import ModelRegistry  # noqa: E402
from db.models.product import Product  # noqa: E402
from db.models.recommendation import RecommendationRun  # noqa: E402
from db.models.shipment import Shipment  # noqa: E402
from db.models.supplier import Supplier  # noqa: E402
from db.models.warehouse import Warehouse  # noqa: E402
from decision_engine.decision_trace import DecisionRequest  # noqa: E402

log = get_logger(__name__)


def print_db_state() -> None:
    with session_scope() as s:
        print("--- DB state ---")
        print(f"  products              : {s.query(Product).count()}")
        print(f"  suppliers             : {s.query(Supplier).count()}")
        print(f"  warehouses            : {s.query(Warehouse).count()}")
        print(f"  model_registry rows   : {s.query(ModelRegistry).count()}")
        print(f"  recommendation_runs   : {s.query(RecommendationRun).count()}")
        print(f"  audit_logs            : {s.query(AuditLog).count()}")
        print(f"  shipments             : {s.query(Shipment).count()}")
        print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="dev")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--service-level", type=float, default=0.95)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warehouse", default=None, help="Override warehouse code (default: first)")
    args = p.parse_args()

    configure_logging()

    print_db_state()

    engine = build_decision_engine(args.mode)

    with session_scope() as s:
        products = s.query(Product).limit(args.n).all()
        warehouses = s.query(Warehouse).all()

        if not products:
            print("No products in DB. Run: python scripts\\ingest_data.py --mode dev")
            return 1
        if not warehouses:
            print("No warehouses in DB. Run: python scripts\\ingest_data.py --mode dev")
            return 1

        if args.warehouse:
            wh = next((w for w in warehouses if w.code == args.warehouse), None)
            if wh is None:
                print(f"Warehouse {args.warehouse} not found.")
                return 1
        else:
            wh = warehouses[0]

        print(f"Using warehouse {wh.code}")
        print(f"Creating up to {args.n} recommendations...\n")

        created = 0
        for i, prod in enumerate(products):
            req = DecisionRequest(
                product_sku=prod.sku,
                warehouse_code=wh.code,
                service_level=args.service_level,
                horizon_days=args.horizon,
                seed=args.seed + i,
            )
            try:
                trace = engine.run(req)
            except Exception as e:  # noqa: BLE001
                print(f"  {prod.sku:16s} FAILED — {type(e).__name__}: {e}")
                continue

            run = RecommendationRun(
                trace_id=trace.trace_id,
                product_id=prod.id,
                warehouse_id=wh.id,
                status=RecommendationStatus.PENDING_REVIEW.value,
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
            s.add(run)
            s.flush()

            s.add(AuditLog(
                actor_id=None,
                actor_name="seed_script",
                action="recommendation.created",
                entity_type="recommendation_run",
                entity_id=run.id,
                before=None,
                after={"status": run.status, "trace_id": run.trace_id},
                trace_id=run.trace_id,
            ))
            s.flush()

            created += 1
            qty = run.final_quantity if run.final_quantity is not None else 0.0
            cost = run.expected_cost if run.expected_cost is not None else 0.0
            sl = run.expected_service_level if run.expected_service_level is not None else 0.0
            print(f"  #{run.id:<4} {prod.sku:16s} qty={qty:8.2f}  cost={cost:10.2f}  service={sl:.4f}")

        print(f"\nCreated {created} recommendation(s).")

    print_db_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())