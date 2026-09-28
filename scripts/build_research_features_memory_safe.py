"""Memory-safe research demand feature builder.

Processes one product at a time and one warehouse series at a time.
This avoids loading the complete 4,565,000-row research demand table
into pandas memory.

The feature definitions themselves remain the project's existing
leakage-safe definitions.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from data_platform.features.temporal_features import (  # noqa: E402
    add_calendar_features,
)
from data_platform.features.demand_features import (  # noqa: E402
    add_all_demand_features,
)


RESEARCH_LAGS = (
    1,
    2,
    3,
    7,
    14,
    21,
    28,
    56,
    91,
    182,
    365,
)

RESEARCH_ROLLING = (
    7,
    14,
    28,
    56,
    91,
    180,
    365,
)

RESEARCH_EXPANDING_MIN = 30


def load_products(input_dir: Path) -> pd.DataFrame:
    """Load the small research products table."""

    products = pd.read_parquet(
        input_dir / "products"
    )

    # Research schema uses `sku`; feature tables use `product_sku`.
    if "sku" in products.columns and "product_sku" not in products.columns:
        products = products.rename(
            columns={"sku": "product_sku"}
        )

    if "product_sku" not in products.columns:
        raise ValueError(
            "Products table must contain either `sku` or `product_sku`. "
            f"Found columns: {products.columns.tolist()}"
        )

    products = products.drop_duplicates(
        subset=["product_sku"]
    ).reset_index(drop=True)

    return products


def create_writer_state(output_path: Path) -> dict:
    return {
        "path": output_path,
        "writer": None,
        "schema": None,
    }


def write_feature_series(
    series_df: pd.DataFrame,
    products: pd.DataFrame,
    writer_state: dict,
) -> int:
    """Build and write features for one product/warehouse series."""

    if series_df.empty:
        return 0

    series_df = series_df.sort_values(
        ["product_sku", "warehouse_code", "demand_date"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Calendar features.
    series_df = add_calendar_features(
        series_df
    )

    # Demand features.
    series_df = add_all_demand_features(
        series_df,
        lags=RESEARCH_LAGS,
        rolling_windows=RESEARCH_ROLLING,
        expanding_min_periods=RESEARCH_EXPANDING_MIN,
        copy=False,
    )

    # Add product metadata without overwriting demand columns.
    product_sku = series_df["product_sku"].iloc[0]

    product_row = products[
        products["product_sku"] == product_sku
    ]

    if not product_row.empty:
        product_row = product_row.iloc[[0]].copy()

        for column in product_row.columns:
            if column == "product_sku":
                continue

            if column not in series_df.columns:
                series_df[column] = product_row.iloc[0][column]

        del product_row

    # Match the project's feature-store convention.
    if "demand_date" in series_df.columns:
        series_df = series_df.rename(
            columns={"demand_date": "date"}
        )

    table = pa.Table.from_pandas(
        series_df,
        preserve_index=False,
    )

    if writer_state["writer"] is None:
        writer_state["schema"] = table.schema

        writer_state["writer"] = pq.ParquetWriter(
            writer_state["path"],
            writer_state["schema"],
            compression="zstd",
            use_dictionary=True,
            write_statistics=False,
        )

    else:
        # Ensure schema remains compatible.
        if table.schema != writer_state["schema"]:
            table = table.cast(
                writer_state["schema"]
            )

    writer_state["writer"].write_table(
        table
    )

    rows = len(series_df)

    del table
    del series_df

    gc.collect()

    return rows


def build_research_demand_features(
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Build demand features without holding the full dataset in RAM."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_table_dir = (
        output_dir / "demand_features"
    )

    output_table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_table_dir / "data.parquet"
    )

    if output_path.exists():
        output_path.unlink()

    # Load only the tiny products table.
    products = load_products(
        input_dir
    )

    # Dataset stays lazy.
    dataset = ds.dataset(
        input_dir / "demand_records",
        format="parquet",
    )

    writer_state = create_writer_state(
        output_path
    )

    total_rows = 0
    total_series = 0

    product_values = (
        products["product_sku"]
        .astype(str)
        .tolist()
    )

    print(
        f"Products to process: {len(product_values):,}"
    )

    try:

        for product_number, product_sku in enumerate(
            product_values,
            start=1,
        ):

            # Read ONLY one product from Arrow.
            #
            # Research dataset:
            # 250 products x 10 warehouses x 1826 days.
            # Therefore one product is roughly 18k rows.
            product_table = dataset.to_table(
                filter=(
                    ds.field("product_sku")
                    == pa.scalar(product_sku)
                ),
                use_threads=False,
            )

            if product_table.num_rows == 0:
                print(
                    f"[{product_number}/{len(product_values)}] "
                    f"{product_sku}: no demand rows"
                )

                del product_table
                continue

            product_df = (
                product_table.to_pandas()
            )

            del product_table

            if product_df.empty:
                del product_df
                continue

            # Make grouping explicit and deterministic.
            product_df = product_df.sort_values(
                [
                    "warehouse_code",
                    "demand_date",
                ],
                kind="mergesort",
            ).reset_index(drop=True)

            warehouse_values = (
                product_df["warehouse_code"]
                .astype(str)
                .unique()
            )

            product_rows = 0

            for warehouse_code in warehouse_values:

                mask = (
                    product_df["warehouse_code"]
                    .astype(str)
                    == warehouse_code
                )

                series_df = (
                    product_df.loc[mask]
                    .copy()
                    .reset_index(drop=True)
                )

                del mask

                if series_df.empty:
                    del series_df
                    continue

                rows = write_feature_series(
                    series_df,
                    products,
                    writer_state,
                )

                product_rows += rows
                total_rows += rows
                total_series += 1

                del series_df

            del product_df

            gc.collect()

            if (
                product_number == 1
                or product_number % 10 == 0
                or product_number == len(product_values)
            ):
                print(
                    f"[{product_number}/{len(product_values)}] "
                    f"{product_sku}: "
                    f"{product_rows:,} rows | "
                    f"total={total_rows:,}"
                )

    finally:

        if writer_state["writer"] is not None:
            writer_state["writer"].close()

    if not output_path.exists():
        raise RuntimeError(
            "Demand feature output was not created."
        )

    manifest = {
        "mode": "research",
        "rows": int(total_rows),
        "series": int(total_series),
        "products": int(len(product_values)),
        "lags": list(RESEARCH_LAGS),
        "rolling_windows": list(RESEARCH_ROLLING),
        "expanding_min_periods": RESEARCH_EXPANDING_MIN,
        "output": str(output_path),
        "compression": "zstd",
        "memory_strategy": (
            "one product at a time, one warehouse series at a time"
        ),
    }

    manifest_path = (
        output_dir
        / "research_demand_stream_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    del products
    del dataset
    gc.collect()

    return manifest


def build_supporting_features(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Build the smaller inventory/supplier feature tables."""

    from data_platform.features.feature_store import (
        build_inventory_features,
        build_supplier_features,
    )

    print()
    print("Building inventory features...")

    build_inventory_features(
        input_dir,
        output_dir,
    )

    print("Building supplier features...")

    build_supplier_features(
        input_dir,
        output_dir,
    )

def main() -> int:

    input_dir = Path(
        "datasets/raw/research"
    )

    output_dir = Path(
        "datasets/curated/research"
    )

    print("=" * 72)
    print(
        "RESEARCH MEMORY-SAFE FEATURE BUILD"
    )
    print("=" * 72)

    result = build_research_demand_features(
        input_dir,
        output_dir,
    )

    print()
    print(
        f"Demand rows:   {result['rows']:,}"
    )
    print(
        f"Demand series: {result['series']:,}"
    )
    print(
        f"Output:        {result['output']}"
    )

    build_supporting_features(
        input_dir,
        output_dir,
    )

    print()
    print("=" * 72)
    print(
        "RESEARCH FEATURE BUILD COMPLETE"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

