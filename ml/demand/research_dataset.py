"""Memory-safe demand dataset builder for the research dataset.

The normal DEV builder remains unchanged.

Research mode:
- never loads the complete demand feature parquet into pandas;
- reads one product at a time through PyArrow;
- selects max_series using the same product/warehouse ordering;
- fits scalers only on training rows;
- creates per-series arrays only for the selected series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from core.logging import get_logger
from data_platform.splits.leakage_detector import assert_no_leakage
from data_platform.splits.time_series_split import chronological_split

from ml.demand.dataset import (
    DATE_COLUMN,
    DECODER_PREFIXES,
    ENCODER_PREFIXES,
    SERIES_KEYS,
    STATIC_NUM_COLUMNS,
    TARGET_COLUMN,
    DemandDatasetBundle,
    DemandScalers,
    SeriesData,
    WindowedDemandDataset,
    _resolve_columns,
)


log = get_logger(__name__)

RESEARCH_SAFE_MAX_SERIES = 100


def _find_columns(
    schema_names: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Resolve encoder/decoder columns from Parquet schema names."""

    encoder_columns = [
        column
        for column in schema_names
        if any(
            column.startswith(prefix)
            for prefix in ENCODER_PREFIXES
        )
    ]

    decoder_columns = [
        column
        for column in schema_names
        if any(
            column.startswith(prefix)
            for prefix in DECODER_PREFIXES
        )
    ]

    if not encoder_columns:
        raise ValueError(
            "No encoder columns found in research feature parquet."
        )

    if not decoder_columns:
        raise ValueError(
            "No decoder columns found in research feature parquet."
        )

    return encoder_columns, decoder_columns


def _read_product(
    dataset: ds.Dataset,
    product_sku: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Read exactly one product from the Arrow dataset."""

    table = dataset.to_table(
        columns=list(columns),
        filter=(
            ds.field("product_sku")
            == pa.scalar(product_sku)
        ),
        use_threads=False,
    )

    if table.num_rows == 0:
        return pd.DataFrame(
            columns=list(columns)
        )

    frame = table.to_pandas()

    del table

    return frame


def _discover_products(
    dataset: ds.Dataset,
) -> List[str]:
    """Discover unique products without loading the whole feature table."""

    products: set[str] = set()

    scanner = dataset.scanner(
        columns=["product_sku"],
        batch_size=65536,
        use_threads=False,
    )

    for batch in scanner.to_batches():

        values = batch.column(0).to_pylist()

        for value in values:
            if value is not None:
                products.add(str(value))

    return sorted(products)


def _discover_selected_series(
    dataset: ds.Dataset,
    products: Sequence[str],
    *,
    max_series: Optional[int],
) -> Dict[str, List[str]]:
    """Select series in the same product/warehouse order as the old builder."""

    selected: Dict[str, List[str]] = {}

    total = 0

    for product in products:

        frame = _read_product(
            dataset,
            product,
            [
                "product_sku",
                "warehouse_code",
            ],
        )

        if frame.empty:
            continue

        warehouses = sorted(
            frame["warehouse_code"]
            .dropna()
            .astype(str)
            .unique()
        )

        chosen: List[str] = []

        for warehouse in warehouses:

            if (
                max_series is not None
                and total >= max_series
            ):
                break

            chosen.append(warehouse)
            total += 1

        if chosen:
            selected[product] = chosen

        del frame

        if (
            max_series is not None
            and total >= max_series
        ):
            break

    return selected


def _get_split_dates(
    dataset: ds.Dataset,
    first_product: str,
    first_warehouse: str,
    *,
    val_fraction: float,
    test_fraction: float,
):
    """Use the existing chronological split implementation on one series.

    All research series share the same global date grid, so the resulting
    cut dates are identical for the selected series.
    """

    sample = _read_product(
        dataset,
        first_product,
        [
            "product_sku",
            "warehouse_code",
            "date",
            "quantity",
        ],
    )

    sample = sample[
        sample["warehouse_code"].astype(str)
        == first_warehouse
    ].copy()

    sample["date"] = pd.to_datetime(
        sample["date"]
    )

    train_sample, val_sample, test_sample, split_dates = (
        chronological_split(
            sample,
            date_col=DATE_COLUMN,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
    )

    # Preserve the project's leakage check.
    assert_no_leakage(
        train_sample,
        val_sample,
        test_sample,
        split_dates,
        date_col=DATE_COLUMN,
    )

    del sample
    del train_sample
    del val_sample
    del test_sample

    return split_dates


def _update_running_stats(
    arr: np.ndarray,
    sums: np.ndarray,
    sums_sq: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Accumulate NaN-safe statistics in float64."""

    if arr.size == 0:
        return

    arr = np.asarray(
        arr,
        dtype=np.float64,
    )

    finite = np.isfinite(arr)

    safe = np.where(
        finite,
        arr,
        0.0,
    )

    sums += safe.sum(
        axis=0,
        dtype=np.float64,
    )

    sums_sq += (
        safe * safe
    ).sum(
        axis=0,
        dtype=np.float64,
    )

    counts += finite.sum(
        axis=0,
        dtype=np.float64,
    )


def _fit_scalers_streaming(
    dataset: ds.Dataset,
    selected_series: Dict[str, List[str]],
    *,
    encoder_columns: List[str],
    split_dates,
):
    """Fit demand scalers using training rows only."""

    columns = [
        "product_sku",
        "warehouse_code",
        "date",
        TARGET_COLUMN,
        *encoder_columns,
    ]

    n_features = len(
        encoder_columns
    )

    sums = np.zeros(
        n_features,
        dtype=np.float64,
    )

    sums_sq = np.zeros(
        n_features,
        dtype=np.float64,
    )

    counts = np.zeros(
        n_features,
        dtype=np.float64,
    )

    target_sum = 0.0
    target_sum_sq = 0.0
    target_count = 0.0

    for product, warehouses in selected_series.items():

        frame = _read_product(
            dataset,
            product,
            columns,
        )

        if frame.empty:
            continue

        frame["date"] = pd.to_datetime(
            frame["date"]
        )

        frame["warehouse_code"] = (
            frame["warehouse_code"]
            .astype(str)
        )

        frame = frame[
            frame["warehouse_code"].isin(
                warehouses
            )
        ]

        train = frame[
            frame["date"]
            < pd.Timestamp(
                split_dates.val_start
            )
        ]

        if not train.empty:

            enc = train[
                encoder_columns
            ].to_numpy(
                dtype=np.float64,
                copy=False,
            )

            _update_running_stats(
                enc,
                sums,
                sums_sq,
                counts,
            )

            target = train[
                TARGET_COLUMN
            ].to_numpy(
                dtype=np.float64,
                copy=False,
            )

            target = np.log1p(
                target
            )

            finite_target = np.isfinite(
                target
            )

            target_safe = target[
                finite_target
            ]

            target_sum += float(
                target_safe.sum(
                    dtype=np.float64
                )
            )

            target_sum_sq += float(
                np.square(
                    target_safe
                ).sum(
                    dtype=np.float64
                )
            )

            target_count += float(
                finite_target.sum()
            )

            del enc
            del target
            del target_safe

        del train
        del frame

    if np.any(counts <= 0):
        bad = [
            encoder_columns[i]
            for i, count in enumerate(counts)
            if count <= 0
        ]

        raise ValueError(
            "Encoder columns have no finite training values: "
            + ", ".join(bad)
        )

    enc_mean = (
        sums / counts
    )

    enc_var = (
        sums_sq / counts
        - enc_mean * enc_mean
    )

    enc_var = np.maximum(
        enc_var,
        0.0,
    )

    enc_std = np.sqrt(
        enc_var
    )

    enc_std = np.where(
        enc_std < 1e-8,
        1.0,
        enc_std,
    )

    if target_count <= 0:
        raise ValueError(
            "No finite training target values found."
        )

    target_mean = (
        target_sum / target_count
    )

    target_var = (
        target_sum_sq / target_count
        - target_mean * target_mean
    )

    target_var = max(
        float(target_var),
        0.0,
    )

    target_std = float(
        np.sqrt(target_var)
    )

    if target_std < 1e-8:
        target_std = 1.0

    return (
        enc_mean.astype(
            np.float32
        ),
        enc_std.astype(
            np.float32
        ),
        float(target_mean),
        float(target_std),
    )


def _build_embeddings(
    dataset: ds.Dataset,
    selected_series: Dict[str, List[str]],
    *,
    split_dates,
):
    """Build category/product/warehouse vocabularies from train data only."""

    columns = [
        "product_sku",
        "warehouse_code",
        "date",
        "category_code",
    ]

    products: set[str] = set()
    categories: set[str] = set()
    warehouses: set[str] = set()

    for product, whs in selected_series.items():

        frame = _read_product(
            dataset,
            product,
            columns,
        )

        if frame.empty:
            continue

        frame["date"] = pd.to_datetime(
            frame["date"]
        )

        frame["warehouse_code"] = (
            frame["warehouse_code"]
            .astype(str)
        )

        frame = frame[
            frame["warehouse_code"].isin(
                whs
            )
        ]

        train = frame[
            frame["date"]
            < pd.Timestamp(
                split_dates.val_start
            )
        ]

        if not train.empty:

            for value in train[
                "product_sku"
            ].dropna().unique():
                products.add(
                    str(value)
                )

            for value in train[
                "category_code"
            ].dropna().unique():
                categories.add(
                    str(value)
                )

            for value in train[
                "warehouse_code"
            ].dropna().unique():
                warehouses.add(
                    str(value)
                )

        del train
        del frame

    product_to_idx = {
        value: i + 1
        for i, value in enumerate(
            sorted(products)
        )
    }

    category_to_idx = {
        value: i + 1
        for i, value in enumerate(
            sorted(categories)
        )
    }

    warehouse_to_idx = {
        value: i + 1
        for i, value in enumerate(
            sorted(warehouses)
        )
    }

    return (
        product_to_idx,
        category_to_idx,
        warehouse_to_idx,
    )


def _make_series(
    frame: pd.DataFrame,
    *,
    encoder_columns: List[str],
    decoder_columns: List[str],
    scalers: DemandScalers,
    product_to_idx: Dict[str, int],
    category_to_idx: Dict[str, int],
    warehouse_to_idx: Dict[str, int],
):
    """Convert one product/warehouse series to SeriesData."""

    frame = frame.sort_values(
        DATE_COLUMN,
        kind="mergesort",
    ).reset_index(drop=True)

    sku = str(
        frame["product_sku"].iloc[0]
    )

    warehouse = str(
        frame["warehouse_code"].iloc[0]
    )

    enc_raw = frame[
        encoder_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    dec_raw = frame[
        decoder_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    target_raw = frame[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    valid = (
        ~np.isnan(enc_raw).any(
            axis=1
        )
        & ~np.isnan(dec_raw).any(
            axis=1
        )
    )

    enc_filled = np.nan_to_num(
        enc_raw,
        nan=0.0,
    )

    dec_filled = np.nan_to_num(
        dec_raw,
        nan=0.0,
    )

    enc_scaled = (
        (
            enc_filled
            - scalers.enc_mean
        )
        / scalers.enc_std
    ).astype(
        np.float32
    )

    log_target = np.log1p(
        target_raw
    )

    target_scaled = (
        (
            log_target
            - scalers.target_mean
        )
        / scalers.target_std
    ).astype(
        np.float32
    )

    category_value = str(
        frame[
            "category_code"
        ].iloc[0]
    )

    static_num = np.array(
        [
            float(
                frame[
                    "unit_cost"
                ].iloc[0]
            ),
            float(
                frame[
                    "unit_price"
                ].iloc[0]
            ),
            float(
                frame[
                    "base_demand"
                ].iloc[0]
            ),
            float(
                frame[
                    "lead_time_days"
                ].iloc[0]
            ),
        ],
        dtype=np.float32,
    )

    series = SeriesData(
        encoder=enc_scaled,
        decoder=dec_filled.astype(
            np.float32
        ),
        target=target_raw.astype(
            np.float32
        ),
        target_scaled=target_scaled,
        static_cat=np.array(
            [
                product_to_idx.get(
                    sku,
                    0,
                ),
                category_to_idx.get(
                    category_value,
                    0,
                ),
                warehouse_to_idx.get(
                    warehouse,
                    0,
                ),
            ],
            dtype=np.int64,
        ),
        static_num=static_num,
        valid_mask=valid,
    )

    dates = pd.to_datetime(
        frame[DATE_COLUMN]
    ).to_numpy()

    del enc_raw
    del dec_raw
    del target_raw
    del enc_filled
    del dec_filled
    del log_target

    return (
        sku,
        warehouse,
        series,
        dates,
    )


def _make_window_indexes(
    series_list: List[SeriesData],
    series_dates: List[np.ndarray],
    *,
    lookback: int,
    horizon: int,
    split_dates,
):
    """Create compact int32 train/val/test window indexes."""

    train_idx: List[Tuple[int, int]] = []
    val_idx: List[Tuple[int, int]] = []
    test_idx: List[Tuple[int, int]] = []

    val_start = pd.Timestamp(
        split_dates.val_start
    )

    test_start = pd.Timestamp(
        split_dates.test_start
    )

    for sid, (
        series,
        dates,
    ) in enumerate(
        zip(
            series_list,
            series_dates,
        )
    ):

        total = len(
            series.target
        )

        for t in range(
            lookback - 1,
            total - horizon,
        ):

            if not series.valid_mask[
                t - lookback + 1
                : t + 1
            ].all():
                continue

            if not series.valid_mask[
                t + 1
                : t + 1 + horizon
            ].all():
                continue

            origin = pd.Timestamp(
                dates[t]
            )

            pair = (
                sid,
                t,
            )

            if origin < val_start:
                train_idx.append(pair)

            elif origin < test_start:
                val_idx.append(pair)

            else:
                test_idx.append(pair)

    return (
        np.asarray(
            train_idx,
            dtype=np.int32,
        ).reshape(-1, 2),
        np.asarray(
            val_idx,
            dtype=np.int32,
        ).reshape(-1, 2),
        np.asarray(
            test_idx,
            dtype=np.int32,
        ).reshape(-1, 2),
    )


def build_demand_datasets_research(
    feature_path: str | Path,
    *,
    lookback: int = 90,
    horizon: int = 30,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    max_series: Optional[int] = None,
) -> DemandDatasetBundle:
    """Memory-safe research dataset builder."""

    feature_path = Path(
        feature_path
    )

    dataset = ds.dataset(
        feature_path,
        format="parquet",
    )

    schema_names = dataset.schema.names

    encoder_columns, decoder_columns = (
        _find_columns(
            schema_names
        )
    )

    products = _discover_products(
        dataset
    )

    if not products:
        raise ValueError(
            "No product_sku values found."
        )

    if max_series is None:
        max_series = RESEARCH_SAFE_MAX_SERIES

        log.warning(
            "research.dataset.safe_cap",
            extra={
                "max_series": max_series,
                "reason": (
                    "8GB-class workstation; "
                    "research builder defaults to "
                    "a memory-safe 100-series subset"
                ),
            },
        )

    selected_series = (
        _discover_selected_series(
            dataset,
            products,
            max_series=max_series,
        )
    )

    total_selected = sum(
        len(v)
        for v in selected_series.values()
    )

    if total_selected == 0:
        raise ValueError(
            "No research series selected."
        )

    log.info(
        "research.dataset.selection",
        extra={
            "products_scanned": len(
                products
            ),
            "selected_series": total_selected,
        },
    )

    first_product = next(
        iter(selected_series)
    )

    first_warehouse = (
        selected_series[
            first_product
        ][0]
    )

    split_dates = _get_split_dates(
        dataset,
        first_product,
        first_warehouse,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    log.info(
        "research.dataset.split",
        extra={
            "val_start": split_dates.val_start.isoformat(),
            "test_start": split_dates.test_start.isoformat(),
        },
    )

    (
        enc_mean,
        enc_std,
        target_mean,
        target_std,
    ) = _fit_scalers_streaming(
        dataset,
        selected_series,
        encoder_columns=encoder_columns,
        split_dates=split_dates,
    )

    (
        product_to_idx,
        category_to_idx,
        warehouse_to_idx,
    ) = _build_embeddings(
        dataset,
        selected_series,
        split_dates=split_dates,
    )

    scalers = DemandScalers(
        enc_mean=enc_mean,
        enc_std=enc_std,
        target_mean=target_mean,
        target_std=target_std,
        encoder_columns=list(
            encoder_columns
        ),
        decoder_columns=list(
            decoder_columns
        ),
        static_num_columns=list(
            STATIC_NUM_COLUMNS
        ),
    )

    required_columns = [
        "product_sku",
        "warehouse_code",
        "date",
        "quantity",
        "category_code",
        "unit_cost",
        "unit_price",
        "base_demand",
        "lead_time_days",
        *encoder_columns,
        *decoder_columns,
    ]

    required_columns = list(
        dict.fromkeys(
            required_columns
        )
    )

    series_arrays: List[SeriesData] = []
    series_dates: List[np.ndarray] = []
    key_to_id: Dict[
        Tuple[str, str],
        int,
    ] = {}

    series_meta: Dict[
        Tuple[str, str],
        dict,
    ] = {}

    for product, warehouses in selected_series.items():

        frame = _read_product(
            dataset,
            product,
            required_columns,
        )

        if frame.empty:
            continue

        frame["warehouse_code"] = (
            frame["warehouse_code"]
            .astype(str)
        )

        frame["date"] = pd.to_datetime(
            frame["date"]
        )

        frame = frame[
            frame["warehouse_code"].isin(
                warehouses
            )
        ].sort_values(
            [
                "warehouse_code",
                "date",
            ],
            kind="mergesort",
        )

        for warehouse, group in frame.groupby(
            "warehouse_code",
            sort=False,
            observed=True,
        ):

            (
                sku,
                wh,
                series,
                dates,
            ) = _make_series(
                group,
                encoder_columns=encoder_columns,
                decoder_columns=decoder_columns,
                scalers=scalers,
                product_to_idx=product_to_idx,
                category_to_idx=category_to_idx,
                warehouse_to_idx=warehouse_to_idx,
            )

            sid = len(
                series_arrays
            )

            series_arrays.append(
                series
            )

            series_dates.append(
                dates
            )

            key_to_id[
                (sku, wh)
            ] = sid

            series_meta[
                (sku, wh)
            ] = {
                "n_rows": len(
                    series.target
                )
            }

        del frame

    (
        train_idx,
        val_idx,
        test_idx,
    ) = _make_window_indexes(
        series_arrays,
        series_dates,
        lookback=lookback,
        horizon=horizon,
        split_dates=split_dates,
    )

    if len(train_idx) == 0:
        raise ValueError(
            "No research training windows produced."
        )

    log.info(
        "research.dataset.windows",
        extra={
            "series": len(
                series_arrays
            ),
            "train": len(
                train_idx
            ),
            "val": len(
                val_idx
            ),
            "test": len(
                test_idx
            ),
        },
    )

    train_ds = WindowedDemandDataset(
        series_arrays,
        train_idx,
        lookback,
        horizon,
    )

    val_ds = WindowedDemandDataset(
        series_arrays,
        val_idx,
        lookback,
        horizon,
    )

    test_ds = WindowedDemandDataset(
        series_arrays,
        test_idx,
        lookback,
        horizon,
    )

    return DemandDatasetBundle(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        scalers=scalers,
        split_dates=split_dates,
        n_products=len(
            product_to_idx
        ) + 1,
        n_categories=len(
            category_to_idx
        ) + 1,
        n_warehouses=len(
            warehouse_to_idx
        ) + 1,
        encoder_columns=list(
            encoder_columns
        ),
        decoder_columns=list(
            decoder_columns
        ),
        static_num_columns=list(
            STATIC_NUM_COLUMNS
        ),
        series_meta=series_meta,
    )
