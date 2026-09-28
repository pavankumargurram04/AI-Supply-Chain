"""CLI: generate a synthetic dataset in one of the predefined modes.

Usage:
    python scripts/generate_synthetic_data.py --mode dev
    python scripts/generate_synthetic_data.py --mode research --seed 42
    python scripts/generate_synthetic_data.py --mode stress --seed 7 --output datasets/raw/stress-7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.deterministic import derive_seed, seed_everything  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402
from data_platform.generator.config import MODE_PRESETS, GeneratorConfig  # noqa: E402
from data_platform.generator.demand import generate_demand  # noqa: E402
from data_platform.generator.entities import (  # noqa: E402
    generate_categories,
    generate_products,
    generate_suppliers,
    generate_warehouses,
)
from data_platform.generator.inventory_flow import generate_inventory_flow  # noqa: E402
from data_platform.generator.supplier_events import generate_supplier_events  # noqa: E402
from data_platform.generator.supplier_product import generate_supplier_products  # noqa: E402
from data_platform.ingestion.parquet_writer import write_manifest, write_table  # noqa: E402

import numpy as np  # noqa: E402

log = get_logger(__name__)


def _rng(seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(seed, *parts))


def generate_all(cfg: GeneratorConfig) -> Dict[str, int]:
    seed_everything(cfg.seed)

    log.info("generator.start", extra={"mode": cfg.mode, "seed": cfg.seed})

    # Master data ------------------------------------------------------------
    categories = generate_categories(cfg, _rng(cfg.seed, "categories"))
    products = generate_products(cfg, categories, _rng(cfg.seed, "products"))
    suppliers = generate_suppliers(cfg, _rng(cfg.seed, "suppliers"))
    warehouses = generate_warehouses(cfg, _rng(cfg.seed, "warehouses"))
    supplier_products = generate_supplier_products(
        cfg, products, suppliers, _rng(cfg.seed, "supplier_products")
    )

    # Demand -----------------------------------------------------------------
    demand_df = generate_demand(cfg, products, warehouses, _rng(cfg.seed, "demand"))

    # Supplier events --------------------------------------------------------
    if cfg.enable_supplier_events:
        supplier_events = generate_supplier_events(
            cfg, suppliers, products, _rng(cfg.seed, "supplier_events")
        )
    else:
        supplier_events = []

    # Inventory ledger -------------------------------------------------------
    if cfg.enable_inventory_ledger:
        movements_df, snapshots_df = generate_inventory_flow(
            cfg, demand_df, supplier_products, _rng(cfg.seed, "inventory")
        )
    else:
        movements_df = pd.DataFrame()
        snapshots_df = pd.DataFrame()

    # Persist ----------------------------------------------------------------
    out = cfg.output_dir
    tables = {
        "categories": pd.DataFrame(categories),
        "products": pd.DataFrame(products),
        "suppliers": pd.DataFrame(suppliers),
        "warehouses": pd.DataFrame(warehouses),
        "supplier_products": pd.DataFrame(supplier_products),
        "supplier_events": pd.DataFrame(supplier_events),
        "demand_records": demand_df,
        "inventory": snapshots_df,
        "inventory_movements": movements_df,
    }

    row_counts: Dict[str, int] = {}
    for name, df in tables.items():
        partition = None
        if name == "demand_records" and not df.empty:
            df = df.copy()
            df["demand_date"] = pd.to_datetime(df["demand_date"])
            df["year"] = df["demand_date"].dt.year
            df["month"] = df["demand_date"].dt.month
            partition = ["year", "month"]
        write_table(df, out, name, partition_by=partition)
        row_counts[name] = int(len(df))

    write_manifest(
        out,
        mode=cfg.mode,
        seed=cfg.seed,
        config=cfg.to_dict(),
        tables=row_counts,
    )

    log.info("generator.done", extra={"mode": cfg.mode, "rows": sum(row_counts.values())})
    return row_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic supply-chain dataset.")
    parser.add_argument("--mode", choices=list(MODE_PRESETS.keys()), default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    configure_logging()

    preset = MODE_PRESETS[args.mode]
    cfg = preset(seed=args.seed) if args.output is None else preset(seed=args.seed, output_dir=args.output)

    counts = generate_all(cfg)
    print("Generated rows per table:")
    for k, v in counts.items():
        print(f"  {k:24s} {v:>12,d}")
    print(f"Total rows: {sum(counts.values()):,d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())