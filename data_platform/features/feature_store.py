"""Materialise features to Parquet.

For every (product_sku, warehouse_code, demand_date) row in
`demand_records`, this module:

1. Adds calendar features (safe, no lag).
2. Adds demand lag/rolling/expanding features (leakage-safe via shift(1)).
3. Adds inventory features from the latest snapshot.
4. Adds supplier features for every supplier linked to the product.

Output is written under `<output_dir>/features/` with:
    demand_features_<feature_version>.parquet
    supplier_features_<feature_version>.parquet
    inventory_features_<feature_version>.parquet

A `_manifest.json` records the FEATURE_VERSION, row counts, and input files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from core.logging import get_logger
from data_platform.features.demand_features import (add_all_demand_features, DEV_LAGS, DEV_ROLLING_WINDOWS, DEV_EXPANDING_MIN_PERIODS)
from data_platform.features.feature_registry import FEATURE_VERSION
from data_platform.features.inventory_features import add_inventory_features
from data_platform.features.supplier_features import add_supplier_features
from data_platform.features.temporal_features import add_calendar_features
from data_platform.ingestion.parquet_writer import read_table

log = get_logger(__name__)


@dataclass
class FeatureBuildResult:
    feature_version: str
    output_dir: str
    row_counts: Dict[str, int]
    manifest_path: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def build_demand_features(
    input_dir: str,
    output_dir: str,
    *,
    mode: Optional[str] = None,
) -> tuple[pd.DataFrame, int]:
    """Build the demand feature table (per product/warehouse/date)."""
    demand = read_table(input_dir, "demand_records")
    if demand.empty:
        return demand, 0

    df = add_calendar_features(demand)

    if mode == "dev":
        df = add_all_demand_features(
            df,
            lags=DEV_LAGS,
            rolling_windows=DEV_ROLLING_WINDOWS,
            expanding_min_periods=DEV_EXPANDING_MIN_PERIODS,
        )
    else:
        df = add_all_demand_features(df, copy=False)

    # Attach product static attributes for embedding / one-hot use downstream.
    products = read_table(input_dir, "products")
    if not products.empty:
        prod_small = products[[
            "sku", "category_code", "unit_cost", "unit_price",
            "lifecycle_stage", "seasonality_class", "volatility_class",
            "base_demand", "lead_time_days",
        ]].rename(columns={"sku": "product_sku"})
        df = df.merge(prod_small, on="product_sku", how="left")

    # Rename product_sku/warehouse_code to stable keys for the store.
    df = df.rename(columns={"demand_date": "date"})

    # Persist.
    out_path = Path(output_dir) / "features" / f"demand_features_{FEATURE_VERSION}.parquet"
    _write_parquet(df, out_path)
    log.info("features.written.demand", extra={"path": str(out_path), "rows": len(df)})
    return df, len(df)


def build_inventory_features(input_dir: str, output_dir: str) -> tuple[pd.DataFrame, int]:
    inv = read_table(input_dir, "inventory")
    if inv.empty:
        return inv, 0
    enriched = add_inventory_features(inv)
    out_path = Path(output_dir) / "features" / f"inventory_features_{FEATURE_VERSION}.parquet"
    _write_parquet(enriched, out_path)
    log.info("features.written.inventory", extra={"path": str(out_path), "rows": len(enriched)})
    return enriched, len(enriched)


def build_supplier_features(
    input_dir: str,
    output_dir: str,
    *,
    demand_features: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, int]:
    """Build per (supplier, product, date) features.

    We use the demand feature table's (product_sku, date) grid as the base
    calendar, expanded across all suppliers linked to each product.
    """
    supplier_products = read_table(input_dir, "supplier_products")
    suppliers = read_table(input_dir, "suppliers")
    events = read_table(input_dir, "supplier_events")

    if supplier_products.empty or suppliers.empty:
        return pd.DataFrame(), 0

    if demand_features is None:
        # Load from disk if the caller didn't pass them.
        p = Path(output_dir) / "features" / f"demand_features_{FEATURE_VERSION}.parquet"
        if not p.exists():
            log.warning("features.supplier.skipped", extra={"reason": "no demand features on disk"})
            return pd.DataFrame(), 0
        demand_features = pd.read_parquet(p)

    # Build a calendar of (product_sku, supplier_code, date) for every link.
    links = supplier_products[["supplier_code", "product_sku"]].drop_duplicates()
    prod_dates = (
        demand_features[["product_sku", "date"]]
        .drop_duplicates()
        .rename(columns={"date": "date"})
    )
    calendar = links.merge(prod_dates, on="product_sku", how="inner")

    features = add_supplier_features(
        supplier_products=supplier_products,
        suppliers=suppliers,
        events=events,
        calendar=calendar,
    )
    out_path = Path(output_dir) / "features" / f"supplier_features_{FEATURE_VERSION}.parquet"
    _write_parquet(features, out_path)
    log.info("features.written.supplier", extra={"path": str(out_path), "rows": len(features)})
    return features, len(features)


def build_all_features(
    input_dir: str,
    output_dir: str,
    *,
    mode: Optional[str] = None,
) -> FeatureBuildResult:
    """Materialise every feature table and write a manifest."""
    log.info("features.build.start", extra={
        "input": input_dir, "output": output_dir, "version": FEATURE_VERSION,
    })

    counts: Dict[str, int] = {}

    demand_df, n = build_demand_features(input_dir, output_dir, mode=mode)
    counts["demand_features"] = n

    inv_df, n = build_inventory_features(input_dir, output_dir)
    counts["inventory_features"] = n

    sup_df, n = build_supplier_features(input_dir, output_dir, demand_features=demand_df)
    counts["supplier_features"] = n

    manifest_path = Path(output_dir) / "features" / "_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "feature_version": FEATURE_VERSION,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "row_counts": counts,
        "total_rows": sum(counts.values()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log.info("features.build.done", extra={"version": FEATURE_VERSION, "rows": sum(counts.values())})

    return FeatureBuildResult(
        feature_version=FEATURE_VERSION,
        output_dir=output_dir,
        row_counts=counts,
        manifest_path=str(manifest_path),
    )


def load_feature_table(output_dir: str, kind: str) -> pd.DataFrame:
    """Load a previously built feature table.

    kind ∈ {"demand", "inventory", "supplier"}.
    """
    p = Path(output_dir) / "features" / f"{kind}_features_{FEATURE_VERSION}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Feature table not found: {p}")
    return pd.read_parquet(p)




