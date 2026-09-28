"""Supplier features.

The original supplier feature implementation is retained for normal/DEV use.

A memory-safe research implementation is also provided. It processes one
product at a time and therefore never loads the 4.565M-row demand feature
table into memory.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _event_kind_flags(events: pd.DataFrame) -> pd.DataFrame:
    """Return daily supplier-level event counts."""

    if events.empty:
        return pd.DataFrame(
            columns=[
                "supplier_code",
                "date",
                "delay_count",
                "quality_count",
                "outage_count",
                "price_change_count",
                "fill_drop_count",
                "delay_days_total",
            ]
        )

    ev = events.copy()

    ev["date"] = pd.to_datetime(
        ev["occurred_at"]
    ).dt.normalize()

    ev["is_delay"] = (
        ev["event_type"] == "delay"
    ).astype("int8")

    ev["is_quality"] = (
        ev["event_type"] == "quality_incident"
    ).astype("int8")

    ev["is_outage"] = (
        ev["event_type"] == "capacity_outage"
    ).astype("int8")

    ev["is_price_change"] = (
        ev["event_type"] == "price_change"
    ).astype("int8")

    ev["is_fill_drop"] = (
        ev["event_type"] == "fill_rate_drop"
    ).astype("int8")

    ev["delay_days"] = ev["payload"].apply(
        lambda p: float(
            p.get("delay_days") or 0.0
        )
        if isinstance(p, dict)
        else 0.0
    )

    daily = (
        ev.groupby(
            ["supplier_code", "date"],
            sort=False,
            observed=True,
        )
        .agg(
            delay_count=("is_delay", "sum"),
            quality_count=("is_quality", "sum"),
            outage_count=("is_outage", "sum"),
            price_change_count=("is_price_change", "sum"),
            fill_drop_count=("is_fill_drop", "sum"),
            delay_days_total=("delay_days", "sum"),
        )
        .reset_index()
    )

    return daily


def _build_daily_supplier_features(
    suppliers: pd.DataFrame,
    events: pd.DataFrame,
    date_range: pd.DatetimeIndex,
    *,
    rolling_windows: tuple[int, ...] = (
        30,
        90,
        180,
        365,
    ),
) -> pd.DataFrame:
    """Build the supplier/date feature table once.

    Research size is only suppliers x days, not suppliers x products x days.
    """

    daily_events = _event_kind_flags(
        events
    )

    supplier_codes = (
        suppliers["code"]
        .dropna()
        .astype(str)
        .unique()
    )

    # ~40 suppliers x 1826 dates in the research dataset.
    daily = pd.MultiIndex.from_product(
        [
            supplier_codes,
            date_range,
        ],
        names=[
            "supplier_code",
            "date",
        ],
    ).to_frame(index=False)

    daily = daily.merge(
        daily_events,
        on=[
            "supplier_code",
            "date",
        ],
        how="left",
        sort=False,
    )

    event_columns = (
        "delay_count",
        "quality_count",
        "outage_count",
        "price_change_count",
        "fill_drop_count",
        "delay_days_total",
    )

    for column in event_columns:
        if column in daily.columns:
            daily[column] = (
                daily[column]
                .fillna(0.0)
                .astype("float32")
            )
        else:
            daily[column] = np.float32(0.0)

    daily.sort_values(
        [
            "supplier_code",
            "date",
        ],
        kind="mergesort",
        inplace=True,
        ignore_index=True,
    )

    group_levels = [
        daily["supplier_code"]
    ]

    # Leakage-safe rolling statistics.
    for window in rolling_windows:

        grouped = daily.groupby(
            "supplier_code",
            sort=False,
            observed=True,
        )

        for column in event_columns:

            lagged = grouped[column].shift(
                1
            )

            rolled = (
                lagged
                .groupby(
                    daily["supplier_code"],
                    sort=False,
                    observed=True,
                )
                .rolling(
                    window,
                    min_periods=1,
                )
                .sum()
                .reset_index(
                    level=0,
                    drop=True,
                )
                .astype("float32")
            )

            daily[
                f"feat_sup_{column}_{window}"
            ] = rolled

            del lagged
            del rolled

        del grouped

        delays = daily[
            f"feat_sup_delay_count_{window}"
        ]

        total_events = (
            daily[
                f"feat_sup_delay_count_{window}"
            ]
            + daily[
                f"feat_sup_quality_count_{window}"
            ]
            + daily[
                f"feat_sup_outage_count_{window}"
            ]
        )

        daily[
            f"feat_sup_ontime_proxy_{window}"
        ] = (
            (
                1.0
                - delays
                / np.maximum(
                    total_events,
                    1.0,
                )
            )
            .clip(
                0.0,
                1.0,
            )
            .astype("float32")
        )

        del delays
        del total_events

    # ------------------------------------------------------------------
    # Recency features.
    # ------------------------------------------------------------------

    for kind, source_column in (
        ("delay", "delay_count"),
        ("quality", "quality_count"),
        ("outage", "outage_count"),
    ):

        event_mask = (
            daily[source_column] > 0
        )

        row_numbers = (
            daily.groupby(
                "supplier_code",
                sort=False,
                observed=True,
            ).cumcount()
        ).astype("int32")

        event_row = row_numbers.where(
            event_mask
        )

        last_event_row = (
            event_row
            .groupby(
                daily["supplier_code"],
                sort=False,
                observed=True,
            )
            .ffill()
        )

        last_event_row = (
            last_event_row
            .groupby(
                daily["supplier_code"],
                sort=False,
                observed=True,
            )
            .shift(1)
        )

        days_since = (
            row_numbers
            - last_event_row
        ).fillna(9999.0)

        daily[
            f"feat_sup_days_since_{kind}"
        ] = days_since.astype(
            "float32"
        )

        del event_mask
        del row_numbers
        del event_row
        del last_event_row
        del days_since

    feature_columns = [
        column
        for column in daily.columns
        if column.startswith("feat_sup_")
    ]

    result = daily[
        [
            "supplier_code",
            "date",
            *feature_columns,
        ]
    ].copy()

    del daily_events
    del daily

    gc.collect()

    return result


def add_supplier_features(
    supplier_products: pd.DataFrame,
    suppliers: pd.DataFrame,
    events: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    rolling_windows: tuple[int, ...] = (
        30,
        90,
        180,
        365,
    ),
) -> pd.DataFrame:
    """Original in-memory supplier feature implementation.

    Kept for existing DEV/test code.
    """

    if calendar.empty:
        return calendar

    daily_events = _event_kind_flags(
        events
    )

    date_range = pd.date_range(
        calendar["date"].min(),
        calendar["date"].max(),
        freq="D",
    )

    supplier_codes = (
        suppliers["code"].unique()
    )

    idx = pd.MultiIndex.from_product(
        [supplier_codes, date_range],
        names=[
            "supplier_code",
            "date",
        ],
    )

    daily = (
        pd.DataFrame(index=idx)
        .reset_index()
    )

    daily = daily.merge(
        daily_events,
        on=[
            "supplier_code",
            "date",
        ],
        how="left",
    )

    for column in (
        "delay_count",
        "quality_count",
        "outage_count",
        "price_change_count",
        "fill_drop_count",
        "delay_days_total",
    ):
        if column in daily.columns:
            daily[column] = (
                daily[column]
                .fillna(0.0)
                .astype("float32")
            )
        else:
            daily[column] = np.float32(
                0.0
            )

    daily = daily.sort_values(
        [
            "supplier_code",
            "date",
        ]
    ).reset_index(drop=True)

    for window in rolling_windows:

        grouped = daily.groupby(
            "supplier_code",
            sort=False,
        )

        for column in (
            "delay_count",
            "quality_count",
            "outage_count",
            "price_change_count",
            "fill_drop_count",
            "delay_days_total",
        ):

            lagged = grouped[column].shift(
                1
            )

            rolled = (
                lagged
                .groupby(
                    daily["supplier_code"],
                    sort=False,
                )
                .rolling(
                    window,
                    min_periods=1,
                )
                .sum()
                .reset_index(
                    level=0,
                    drop=True,
                )
            )

            daily[
                f"feat_sup_{column}_{window}"
            ] = rolled.astype(
                "float32"
            )

        delays = daily[
            f"feat_sup_delay_count_{window}"
        ]

        total_events = (
            daily[
                f"feat_sup_delay_count_{window}"
            ]
            + daily[
                f"feat_sup_quality_count_{window}"
            ]
            + daily[
                f"feat_sup_outage_count_{window}"
            ]
        )

        daily[
            f"feat_sup_ontime_proxy_{window}"
        ] = (
            (
                1.0
                - delays
                / np.maximum(
                    total_events,
                    1.0,
                )
            )
            .clip(
                0.0,
                1.0,
            )
            .astype("float32")
        )

    for kind, column in (
        ("delay", "delay_count"),
        ("quality", "quality_count"),
        ("outage", "outage_count"),
    ):

        temp = daily[
            [
                "supplier_code",
                "date",
                column,
            ]
        ].copy()

        temp["is_event"] = (
            temp[column] > 0
        ).astype("int8")

        temp = temp.sort_values(
            [
                "supplier_code",
                "date",
            ]
        )

        temp["row"] = np.arange(
            len(temp)
        )

        grouped = temp.groupby(
            "supplier_code",
            sort=False,
        )

        temp["event_row"] = temp[
            "row"
        ].where(
            temp["is_event"] == 1
        )

        last_event_row = (
            temp.groupby(
                "supplier_code",
                sort=False,
            )["event_row"]
            .ffill()
        )

        last_event_row = (
            last_event_row
            .groupby(
                temp["supplier_code"]
            )
            .shift(1)
        )

        days_since = (
            temp["row"]
            - last_event_row
        ).fillna(9999.0)

        daily[
            f"feat_sup_days_since_{kind}"
        ] = days_since.astype(
            "float32"
        ).values

    feature_cols = [
        column
        for column in daily.columns
        if column.startswith("feat_sup_")
    ]

    daily_small = daily[
        [
            "supplier_code",
            "date",
            *feature_cols,
        ]
    ].copy()

    calendar = calendar.copy()

    calendar = calendar.merge(
        daily_small,
        on=[
            "supplier_code",
            "date",
        ],
        how="left",
        suffixes=("", "_sup"),
    )

    static = suppliers[
        [
            "code",
            "reliability_class",
            "cost_class",
            "quality_class",
            "lead_time_class",
            "capacity_class",
        ]
    ].rename(
        columns={
            "code": "supplier_code",
            "reliability_class":
                "feat_sup_reliability",
            "cost_class":
                "feat_sup_cost_class",
            "quality_class":
                "feat_sup_quality_class",
            "lead_time_class":
                "feat_sup_lead_time_class",
            "capacity_class":
                "feat_sup_capacity_class",
        }
    )

    calendar = calendar.merge(
        static,
        on="supplier_code",
        how="left",
    )

    link_cols = [
        "supplier_code",
        "product_sku",
        "lead_time_days",
        "lead_time_std_days",
        "unit_price",
        "moq",
        "shipping_cost",
    ]

    links = supplier_products[
        link_cols
    ].rename(
        columns={
            "lead_time_days":
                "feat_sup_lead_time_days",
            "lead_time_std_days":
                "feat_sup_lead_time_std_days",
            "unit_price":
                "feat_sup_unit_price",
            "moq":
                "feat_sup_moq",
            "shipping_cost":
                "feat_sup_shipping_cost",
        }
    )

    calendar = calendar.merge(
        links,
        on=[
            "supplier_code",
            "product_sku",
        ],
        how="left",
    )

    return calendar


def _read_research_date_range(
    input_dir: Path,
    first_product: str,
) -> pd.DatetimeIndex:
    """Read only one product to determine the research date grid."""

    dataset = ds.dataset(
        input_dir / "demand_records",
        format="parquet",
    )

    table = dataset.to_table(
        columns=[
            "product_sku",
            "demand_date",
        ],
        filter=(
            ds.field("product_sku")
            == pa.scalar(first_product)
        ),
        use_threads=False,
    )

    dates = table.column(
        "demand_date"
    ).to_pandas()

    dates = pd.to_datetime(
        dates
    ).dt.normalize()

    min_date = dates.min()
    max_date = dates.max()

    del table
    del dates
    del dataset

    return pd.date_range(
        min_date,
        max_date,
        freq="D",
    )


def build_supplier_features_memory_safe(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    feature_version: str,
) -> tuple[int, int]:
    """Build research supplier features product-by-product.

    Returns:
        (row_count, product_count)
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    supplier_products = pd.read_parquet(
        input_dir / "supplier_products"
    )

    suppliers = pd.read_parquet(
        input_dir / "suppliers"
    )

    events = pd.read_parquet(
        input_dir / "supplier_events"
    )

    required_link_columns = [
        "supplier_code",
        "product_sku",
        "lead_time_days",
        "lead_time_std_days",
        "unit_price",
        "moq",
        "shipping_cost",
    ]

    missing = [
        column
        for column in required_link_columns
        if column not in supplier_products.columns
    ]

    if missing:
        raise ValueError(
            "supplier_products missing columns: "
            + ", ".join(missing)
        )

    if supplier_products.empty:
        return 0, 0

    output_dir = Path(output_dir)
    features_dir = (
        output_dir / "features"
    )

    features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        features_dir
        / f"supplier_features_{feature_version}.parquet"
    )

    if output_path.exists():
        output_path.unlink()

    product_values = (
        supplier_products[
            "product_sku"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if not product_values:
        return 0, 0

    date_range = _read_research_date_range(
        input_dir,
        product_values[0],
    )

    daily_features = _build_daily_supplier_features(
        suppliers,
        events,
        date_range,
    )

    static = suppliers[
        [
            "code",
            "reliability_class",
            "cost_class",
            "quality_class",
            "lead_time_class",
            "capacity_class",
        ]
    ].rename(
        columns={
            "code":
                "supplier_code",
            "reliability_class":
                "feat_sup_reliability",
            "cost_class":
                "feat_sup_cost_class",
            "quality_class":
                "feat_sup_quality_class",
            "lead_time_class":
                "feat_sup_lead_time_class",
            "capacity_class":
                "feat_sup_capacity_class",
        }
    )

    writer = None
    schema = None

    total_rows = 0
    processed_products = 0

    try:

        for index, product_sku in enumerate(
            product_values,
            start=1,
        ):

            links = (
                supplier_products[
                    supplier_products[
                        "product_sku"
                    ].astype(str)
                    == product_sku
                ][
                    required_link_columns
                ]
                .drop_duplicates(
                    [
                        "supplier_code",
                        "product_sku",
                    ]
                )
                .copy()
            )

            if links.empty:
                continue

            supplier_codes = (
                links[
                    "supplier_code"
                ]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            # One product only:
            # suppliers_for_product x dates.
            calendar = pd.MultiIndex.from_product(
                [
                    supplier_codes,
                    date_range,
                ],
                names=[
                    "supplier_code",
                    "date",
                ],
            ).to_frame(
                index=False
            )

            calendar[
                "product_sku"
            ] = product_sku

            # Supplier historical features.
            calendar = calendar.merge(
                daily_features,
                on=[
                    "supplier_code",
                    "date",
                ],
                how="left",
                sort=False,
            )

            # Supplier static attributes.
            calendar = calendar.merge(
                static,
                on="supplier_code",
                how="left",
                sort=False,
            )

            # Product-specific supplier terms.
            link_features = links.rename(
                columns={
                    "lead_time_days":
                        "feat_sup_lead_time_days",
                    "lead_time_std_days":
                        "feat_sup_lead_time_std_days",
                    "unit_price":
                        "feat_sup_unit_price",
                    "moq":
                        "feat_sup_moq",
                    "shipping_cost":
                        "feat_sup_shipping_cost",
                }
            )

            calendar = calendar.merge(
                link_features,
                on=[
                    "supplier_code",
                    "product_sku",
                ],
                how="left",
                sort=False,
            )

            table = pa.Table.from_pandas(
                calendar,
                preserve_index=False,
            )

            if writer is None:
                schema = table.schema

                writer = pq.ParquetWriter(
                    output_path,
                    schema,
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=False,
                )

            elif table.schema != schema:
                table = table.cast(
                    schema
                )

            writer.write_table(
                table
            )

            rows = len(calendar)

            total_rows += rows
            processed_products += 1

            del table
            del calendar
            del links
            del link_features

            gc.collect()

            if (
                index == 1
                or index % 10 == 0
                or index == len(product_values)
            ):
                print(
                    f"Supplier [{index}/{len(product_values)}] "
                    f"{product_sku}: "
                    f"{rows:,} rows | "
                    f"total={total_rows:,}"
                )

    finally:

        if writer is not None:
            writer.close()

    del daily_features
    del static
    del supplier_products
    del suppliers
    del events

    gc.collect()

    if not output_path.exists():
        raise RuntimeError(
            "Supplier feature parquet was not created."
        )

    return total_rows, processed_products
