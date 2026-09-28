"""Lag and rolling features from demand history.

CRITICAL: everything here is shift(1) so that the feature at date t only
uses values strictly before t. This is the single most important guard
against target leakage.

The full feature pipeline is memory-conscious:
- the dataframe is sorted only once;
- intermediate feature functions can operate on the same dataframe;
- repeated full dataframe copies are avoided;
- categorical grouping columns are supported efficiently.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_LAGS: Sequence[int] = (
    1, 2, 3, 7, 14, 21, 28, 56, 91, 182, 365
)

DEFAULT_ROLLING_WINDOWS: Sequence[int] = (
    7, 14, 28, 56, 91, 180, 365
)

DEFAULT_EXPANDING_MIN_PERIODS = 30

DEV_LAGS: Sequence[int] = (
    1, 2, 3, 7, 14, 21, 28, 56, 91
)

DEV_ROLLING_WINDOWS: Sequence[int] = (
    7, 14, 21, 28, 56, 91
)

DEV_EXPANDING_MIN_PERIODS = 30


def _prepare_frame(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    copy: bool,
    sort: bool,
) -> pd.DataFrame:
    """Prepare a dataframe while allowing the full pipeline to reuse memory."""

    if copy:
        out = df.copy()
    else:
        out = df

    if sort:
        sort_cols = list(group_cols) + ["demand_date"]

        if copy:
            out = out.sort_values(
                sort_cols,
                kind="mergesort",
            ).reset_index(drop=True)
        else:
            out.sort_values(
                sort_cols,
                kind="mergesort",
                inplace=True,
                ignore_index=True,
            )

    return out


def _groupby(
    df: pd.DataFrame,
    group_cols: Sequence[str],
):
    """Memory-efficient groupby helper."""

    return df.groupby(
        list(group_cols),
        sort=False,
        observed=True,
        dropna=False,
    )


def add_demand_lags(
    df: pd.DataFrame,
    *,
    value_col: str = "quantity",
    group_cols: Sequence[str] = ("product_sku", "warehouse_code"),
    lags: Iterable[int] = DEFAULT_LAGS,
    copy: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Add leakage-safe lag features."""

    out = _prepare_frame(
        df,
        group_cols=group_cols,
        copy=copy,
        sort=sort,
    )

    grouped = _groupby(out, group_cols)[value_col]

    for lag in lags:
        feature = grouped.shift(lag).astype("float32")
        out[f"feat_demand_lag_{lag}"] = feature
        del feature

    del grouped

    return out


def add_demand_rolling(
    df: pd.DataFrame,
    *,
    value_col: str = "quantity",
    group_cols: Sequence[str] = ("product_sku", "warehouse_code"),
    windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    copy: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Add leakage-safe rolling demand statistics."""

    out = _prepare_frame(
        df,
        group_cols=group_cols,
        copy=copy,
        sort=sort,
    )

    # Shift first so date t never sees quantity at t.
    lagged = _groupby(out, group_cols)[value_col].shift(1)

    grouped = lagged.groupby(
        [out[c] for c in group_cols],
        sort=False,
        observed=True,
        dropna=False,
    )

    group_levels = list(range(len(group_cols)))

    for window in windows:
        min_periods = max(1, window // 2)

        rolling_obj = grouped.rolling(
            window,
            min_periods=min_periods,
        )

        feature = (
            rolling_obj.mean()
            .reset_index(
                level=group_levels,
                drop=True,
            )
            .astype("float32")
        )
        out[f"feat_demand_roll_mean_{window}"] = feature
        del feature
        del rolling_obj

        rolling_obj = grouped.rolling(
            window,
            min_periods=min_periods,
        )

        feature = (
            rolling_obj.std()
            .reset_index(
                level=group_levels,
                drop=True,
            )
            .astype("float32")
        )
        out[f"feat_demand_roll_std_{window}"] = feature
        del feature
        del rolling_obj

        rolling_obj = grouped.rolling(
            window,
            min_periods=min_periods,
        )

        feature = (
            rolling_obj.max()
            .reset_index(
                level=group_levels,
                drop=True,
            )
            .astype("float32")
        )
        out[f"feat_demand_roll_max_{window}"] = feature
        del feature
        del rolling_obj

        rolling_obj = grouped.rolling(
            window,
            min_periods=min_periods,
        )

        feature = (
            rolling_obj.min()
            .reset_index(
                level=group_levels,
                drop=True,
            )
            .astype("float32")
        )
        out[f"feat_demand_roll_min_{window}"] = feature
        del feature
        del rolling_obj

    del lagged
    del grouped

    return out


def add_demand_expanding(
    df: pd.DataFrame,
    *,
    value_col: str = "quantity",
    group_cols: Sequence[str] = ("product_sku", "warehouse_code"),
    min_periods: int = DEFAULT_EXPANDING_MIN_PERIODS,
    copy: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Add leakage-safe expanding demand statistics."""

    out = _prepare_frame(
        df,
        group_cols=group_cols,
        copy=copy,
        sort=sort,
    )

    lagged = _groupby(out, group_cols)[value_col].shift(1)

    grouped = lagged.groupby(
        [out[c] for c in group_cols],
        sort=False,
        observed=True,
        dropna=False,
    )

    group_levels = list(range(len(group_cols)))

    feature = (
        grouped.expanding(min_periods=min_periods)
        .mean()
        .reset_index(
            level=group_levels,
            drop=True,
        )
        .astype("float32")
    )

    out["feat_demand_expand_mean"] = feature
    del feature

    feature = (
        grouped.expanding(min_periods=min_periods)
        .std()
        .reset_index(
            level=group_levels,
            drop=True,
        )
        .astype("float32")
    )

    out["feat_demand_expand_std"] = feature
    del feature

    del lagged
    del grouped

    return out


def add_promotion_history(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("product_sku", "warehouse_code"),
    copy: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Add lagged 7-day and 28-day promotion/holiday counts."""

    out = _prepare_frame(
        df,
        group_cols=group_cols,
        copy=copy,
        sort=sort,
    )

    grouped = _groupby(out, group_cols)

    promo_lag = (
        grouped["promotion_flag"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    hol_lag = (
        grouped["holiday_flag"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    gidx = [out[c] for c in group_cols]
    group_levels = list(range(len(group_cols)))

    promo_grouped = promo_lag.groupby(
        gidx,
        sort=False,
        observed=True,
        dropna=False,
    )

    feature = (
        promo_grouped.rolling(7, min_periods=1)
        .sum()
        .reset_index(
            level=group_levels,
            drop=True,
        )
        .astype("int8")
    )
    out["feat_promo_count_7"] = feature
    del feature

    feature = (
        promo_grouped.rolling(28, min_periods=1)
        .sum()
        .reset_index(
            level=group_levels,
            drop=True,
        )
        .astype("int8")
    )
    out["feat_promo_count_28"] = feature
    del feature

    hol_grouped = hol_lag.groupby(
        gidx,
        sort=False,
        observed=True,
        dropna=False,
    )

    feature = (
        hol_grouped.rolling(7, min_periods=1)
        .sum()
        .reset_index(
            level=group_levels,
            drop=True,
        )
        .astype("int8")
    )
    out["feat_holiday_count_7"] = feature
    del feature

    feature = (
        hol_grouped.rolling(28, min_periods=1)
        .sum()
        .reset_index(
            level=group_levels,
            drop=True,
        )
        .astype("int8")
    )
    out["feat_holiday_count_28"] = feature
    del feature

    del grouped
    del promo_lag
    del hol_lag
    del promo_grouped
    del hol_grouped

    return out


def add_censored_features(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("product_sku", "warehouse_code"),
    copy: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Add lagged stockout/censored-day counts."""

    out = _prepare_frame(
        df,
        group_cols=group_cols,
        copy=copy,
        sort=sort,
    )

    if "is_censored" not in out.columns:
        out["feat_censored_count_7"] = np.int8(0)
        out["feat_censored_count_28"] = np.int8(0)
        return out

    cens = (
        _groupby(out, group_cols)["is_censored"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    gidx = [out[c] for c in group_cols]
    group_levels = list(range(len(group_cols)))

    cens_grouped = cens.groupby(
        gidx,
        sort=False,
        observed=True,
        dropna=False,
    )

    feature = (
        cens_grouped.rolling(7, min_periods=1)
        .sum()
        .reset_index(
            level=group_levels,
            drop=True,
        )
        .astype("int8")
    )

    out["feat_censored_count_7"] = feature
    del feature

    feature = (
        cens_grouped.rolling(28, min_periods=1)
        .sum()
        .reset_index(
            level=group_levels,
            drop=True,
        )
        .astype("int8")
    )

    out["feat_censored_count_28"] = feature
    del feature

    del cens
    del cens_grouped

    return out


def add_all_demand_features(
    df: pd.DataFrame,
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    expanding_min_periods: int = DEFAULT_EXPANDING_MIN_PERIODS,
    copy: bool = True,
) -> pd.DataFrame:
    """Run the complete demand feature pipeline.

    The dataframe is sorted exactly once.  When copy=False, the supplied
    dataframe is reused and mutated in place, which is required for the
    large research dataset.
    """

    group_cols = (
        "product_sku",
        "warehouse_code",
    )

    out = df.copy() if copy else df

    # Sort exactly once.
    if copy:
        out = out.sort_values(
            list(group_cols) + ["demand_date"],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        out.sort_values(
            list(group_cols) + ["demand_date"],
            kind="mergesort",
            inplace=True,
            ignore_index=True,
        )

    # All subsequent functions reuse the same dataframe.
    out = add_demand_lags(
        out,
        lags=lags,
        group_cols=group_cols,
        copy=False,
        sort=False,
    )

    out = add_demand_rolling(
        out,
        windows=rolling_windows,
        group_cols=group_cols,
        copy=False,
        sort=False,
    )

    out = add_demand_expanding(
        out,
        min_periods=expanding_min_periods,
        group_cols=group_cols,
        copy=False,
        sort=False,
    )

    out = add_promotion_history(
        out,
        group_cols=group_cols,
        copy=False,
        sort=False,
    )

    out = add_censored_features(
        out,
        group_cols=group_cols,
        copy=False,
        sort=False,
    )

    return out
