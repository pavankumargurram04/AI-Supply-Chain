"""CLI: load a previously generated Parquet dataset into PostgreSQL.

Usage:
    python scripts/ingest_data.py --mode dev
    python scripts/ingest_data.py --input datasets/raw/research
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database import session_scope  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402
from data_platform.ingestion.db_ingestor import (  # noqa: E402
    ingest_categories,
    ingest_demand,
    ingest_inventory,
    ingest_inventory_movements,
    ingest_products,
    ingest_supplier_events,
    ingest_supplier_products,
    ingest_suppliers,
    ingest_warehouses,
)
from data_platform.ingestion.parquet_writer import read_table  # noqa: E402

log = get_logger(__name__)


_INGEST_ORDER = [
    ("categories", ingest_categories),
    ("suppliers", ingest_suppliers),
    ("warehouses", ingest_warehouses),
    ("products", ingest_products),         # after categories
    ("supplier_products", ingest_supplier_products),  # after suppliers + products
    ("supplier_events", ingest_supplier_events),
    ("demand_records", ingest_demand),
    ("inventory", ingest_inventory),
    ("inventory_movements", ingest_inventory_movements),
]


def ingest_all(input_dir: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table_name, fn in _INGEST_ORDER:
        df = read_table(input_dir, table_name)
        if df.empty:
            counts[table_name] = 0
            continue
        with session_scope() as session:
            inserted = fn(session, df)
        counts[table_name] = inserted
        log.info("ingest.table", extra={"table": table_name, "inserted": inserted})
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest generated dataset into PostgreSQL.")
    parser.add_argument("--mode", choices=["dev", "research", "stress"], default=None,
                        help="Convenience shortcut for standard dataset paths.")
    parser.add_argument("--input", type=str, default=None,
                        help="Explicit input directory (overrides --mode).")
    args = parser.parse_args()

    configure_logging()

    if args.input is None and args.mode is None:
        parser.error("Provide either --mode or --input.")

    input_dir = args.input or f"datasets/raw/{args.mode}"

    if not Path(input_dir).exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    counts = ingest_all(input_dir)
    print("Inserted rows per table:")
    for k, v in counts.items():
        print(f"  {k:24s} {v:>12,d}")
    print(f"Total: {sum(counts.values()):,d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())