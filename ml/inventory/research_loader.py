"""Memory-safe inventory series loader for the research dataset."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def _find_demand_feature_file(
    features_dir: Path,
) -> Path:
    files = sorted(
        features_dir.glob(
            "demand_features_*.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No demand_features_*.parquet in {features_dir}"
        )

    return files[-1]


def _discover_series(
    dataset: ds.Dataset,
    max_series: int,
) -> List[Tuple[str, str]]:
    """Discover the first max_series (product, warehouse) pairs."""

    seen = set()
    result = []

    scanner = dataset.scanner(
        columns=[
            "product_sku",
            "warehouse_code",
        ],
        batch_size=65536,
        use_threads=False,
    )

    for batch in scanner.to_batches():

        skus = batch.column(
            "product_sku"
        ).to_pylist()

        warehouses = batch.column(
            "warehouse_code"
        ).to_pylist()

        for sku, wh in zip(
            skus,
            warehouses,
        ):
            if sku is None or wh is None:
                continue

            key = (
                str(sku),
                str(wh),
            )

            if key not in seen:
                seen.add(key)
                result.append(key)

                if len(result) >= max_series:
                    return result

    return result


def _read_series(
    dataset: ds.Dataset,
    sku: str,
    warehouse: str,
) -> pd.DataFrame:
    """Read one product/warehouse demand series."""

    table = dataset.to_table(
        columns=[
            "product_sku",
            "warehouse_code",
            "date",
            "quantity",
            "unit_cost",
            "feat_demand_lag_1",
            "feat_demand_roll_std_28",
        ],
        filter=(
            (ds.field("product_sku") == pa.scalar(sku))
            & (
                ds.field("warehouse_code")
                == pa.scalar(warehouse)
            )
        ),
        use_threads=False,
    )

    if table.num_rows == 0:
        return pd.DataFrame()

    df = table.to_pandas()

    del table

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df.sort_values(
        "date",
        kind="mergesort",
        inplace=True,
        ignore_index=True,
    )

    return df


def _lead_time_summary(
    input_dir: Path,
) -> pd.DataFrame:
    """Compute product-level supplier lead-time statistics.

    Uses the small raw supplier_products table instead of the huge
    supplier feature table.
    """

    supplier_products = pd.read_parquet(
        input_dir / "supplier_products",
        columns=[
            "product_sku",
            "lead_time_days",
            "lead_time_std_days",
        ],
    )

    summary = (
        supplier_products
        .groupby(
            "product_sku",
            sort=False,
            observed=True,
        )
        .agg(
            lead_time_mean=(
                "lead_time_days",
                "mean",
            ),
            lead_time_std=(
                "lead_time_std_days",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


def load_research_inventory_series(
    features_dir: str | Path,
    *,
    input_dir: str | Path = "datasets/raw/research",
    max_series: int = 10,
    seed: int = 42,
) -> List[dict]:
    """Return only max_series inventory series."""

    features_dir = Path(
        features_dir
    )

    input_dir = Path(
        input_dir
    )

    feature_path = _find_demand_feature_file(
        features_dir
    )

    dataset = ds.dataset(
        feature_path,
        format="parquet",
    )

    keys = _discover_series(
        dataset,
        max_series=max_series,
    )

    if not keys:
        raise ValueError(
            "No research inventory series found."
        )

    lead_times = _lead_time_summary(
        input_dir
    )

    series_list = []

    for index, (sku, wh) in enumerate(
        keys,
        start=1,
    ):

        df = _read_series(
            dataset,
            sku,
            wh,
        )

        if df.empty:
            continue

        T = len(df)

        if T < 30:
            del df
            continue

        # Same documented design as the existing trainer:
        # lag-1 demand is the forecast proxy.
        fc = (
            df["feat_demand_lag_1"]
            .ffill()
            .bfill()
            .fillna(0.0)
            .to_numpy(
                dtype=np.float32
            )
        )

        if (
            "feat_demand_roll_std_28"
            in df.columns
        ):
            fc_std = (
                df[
                    "feat_demand_roll_std_28"
                ]
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float32
                )
            )
        else:
            fc_std = (
                0.15 * fc
            ).astype(
                np.float32
            )

        actual = (
            df["quantity"]
            .to_numpy(
                dtype=np.float32
            )
        )

        lead = lead_times[
            lead_times[
                "product_sku"
            ].astype(str)
            == str(sku)
        ]

        if not lead.empty:
            ltm = float(
                lead[
                    "lead_time_mean"
                ].iloc[0]
            )
            lts = float(
                lead[
                    "lead_time_std"
                ].iloc[0]
            )
        else:
            ltm = 7.0
            lts = 1.5

        lt_mean = np.full(
            T,
            ltm,
            dtype=np.float32,
        )

        lt_std = np.full(
            T,
            lts,
            dtype=np.float32,
        )

        # Research inventory run currently uses the documented fallback risk.
        # The supplier-risk model can be integrated later without changing
        # the series-loading architecture.
        risk = np.full(
            T,
            0.1,
            dtype=np.float32,
        )

        unit_cost = float(
            df[
                "unit_cost"
            ].iloc[0]
        )

        series_list.append(
            {
                "product_sku": sku,
                "warehouse_code": wh,
                "forecast_point": fc,
                "forecast_std": fc_std,
                "actual_demand": actual,
                "lead_time_mean": lt_mean,
                "lead_time_std": lt_std,
                "supplier_risk": risk,
                "unit_cost": unit_cost,
            }
        )

        del df
        del fc
        del fc_std
        del actual
        del lt_mean
        del lt_std
        del risk

        gc.collect()

        if (
            index == 1
            or index % 5 == 0
            or index == len(keys)
        ):
            print(
                f"Research inventory series "
                f"[{index}/{len(keys)}] "
                f"{sku}:{wh} "
                f"rows={T:,}"
            )

    del lead_times
    del dataset

    gc.collect()

    if not series_list:
        raise ValueError(
            "No usable research inventory series."
        )

    return series_list
