"""CLI: run every monitor and persist alerts.

Usage:
    python scripts\run_monitoring.py --mode dev
    python scripts\run_monitoring.py --mode dev --no-persist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logging import configure_logging, get_logger  # noqa: E402
from monitoring.alerting import write_alerts  # noqa: E402
from monitoring.forecast_monitor import monitor_demand  # noqa: E402
from monitoring.inventory_monitor import monitor_inventory  # noqa: E402
from monitoring.supplier_monitor import monitor_supplier  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description="Run all monitoring checks.")
    p.add_argument("--mode", choices=["dev", "research", "stress"], default="dev")
    p.add_argument("--no-persist", action="store_true",
                   help="Do not write alerts to the DB; just print the report.")
    p.add_argument("--recent-days", type=int, default=30)
    p.add_argument("--reference-days", type=int, default=180)
    args = p.parse_args()

    configure_logging()

    features_dir = f"datasets/curated/{args.mode}/features"
    movements_path = None  # auto-discovered

    reports = []
    reports.append(monitor_demand(
        features_dir=features_dir,
        recent_days=args.recent_days,
        reference_days=args.reference_days,
    ))
    reports.append(monitor_supplier(
        features_dir=features_dir,
        recent_days=args.recent_days,
        reference_days=args.reference_days,
    ))
    reports.append(monitor_inventory(
        movements_path=movements_path,
        recent_days=args.recent_days,
        reference_days=args.reference_days,
    ))

    summary = {
        "mode": args.mode,
        "reports": [r.to_dict() for r in reports],
        "total_alerts": sum(len(r.alerts) for r in reports),
    }

    out_dir = Path("reports/monitoring")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"monitoring_{args.mode}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if not args.no_persist:
        try:
            from core.database import session_scope
            with session_scope() as session:
                for r in reports:
                    write_alerts(session, r)
        except Exception as e:  # noqa: BLE001
            log.warning("monitoring.persist_failed", extra={"error": str(e)})

    # Print.
    print(f"Monitoring report written to: {out_path}")
    for r in reports:
        print(f"\n== {r.name} ==")
        for k, v in r.metrics.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")
        print(f"  alerts: {len(r.alerts)}")
        for a in r.alerts:
            print(f"    [{a.severity.upper()}] {a.title}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())