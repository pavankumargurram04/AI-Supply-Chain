"""Calendar and time-derived features.

These features are pure functions of the calendar date and therefore cannot
leak future target values.  The implementation is memory-conscious so it can
operate on the large research dataset without creating unnecessary temporary
float64 arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Major holiday dates (month, day) -- matching the generator's holiday set.
MAJOR_HOLIDAYS = [
    (1, 1),
    (5, 1),
    (7, 4),
    (11, 25),
    (12, 24),
    (12, 25),
    (12, 31),
]


def add_calendar_features(
    df: pd.DataFrame,
    *,
    date_col: str = "demand_date",
) -> pd.DataFrame:
    """Add calendar features using memory-conscious operations."""

    out = df.copy()

    # Keep datetime representation once.
    d = pd.to_datetime(out[date_col])

    # ------------------------------------------------------------------
    # Basic calendar features
    # ------------------------------------------------------------------

    dow = d.dt.weekday.to_numpy(dtype=np.int8)
    dom = d.dt.day.to_numpy(dtype=np.int8)
    month = d.dt.month.to_numpy(dtype=np.int8)
    quarter = d.dt.quarter.to_numpy(dtype=np.int8)
    year = d.dt.year.to_numpy(dtype=np.int16)
    week = d.dt.isocalendar().week.to_numpy(dtype=np.int8)
    doy = d.dt.dayofyear.to_numpy(dtype=np.int16)

    out["feat_dow"] = dow
    out["feat_dom"] = dom
    out["feat_month"] = month
    out["feat_quarter"] = quarter
    out["feat_year"] = year
    out["feat_week"] = week
    out["feat_day_of_year"] = doy

    # ------------------------------------------------------------------
    # Cyclical day-of-week features
    #
    # Compute one angle and immediately discard it after both features.
    # ------------------------------------------------------------------

    dow_angle = (
        np.float32(2.0 * np.pi / 7.0)
        * dow.astype(np.float32)
    )

    out["feat_dow_sin"] = np.sin(dow_angle).astype(np.float32)
    out["feat_dow_cos"] = np.cos(dow_angle).astype(np.float32)

    del dow_angle

    # ------------------------------------------------------------------
    # Cyclical month features
    # ------------------------------------------------------------------

    month_zero = month.astype(np.float32) - np.float32(1.0)

    month_angle = (
        np.float32(2.0 * np.pi / 12.0)
        * month_zero
    )

    out["feat_month_sin"] = np.sin(month_angle).astype(np.float32)
    out["feat_month_cos"] = np.cos(month_angle).astype(np.float32)

    del month_zero
    del month_angle

    # ------------------------------------------------------------------
    # Cyclical day-of-year features
    #
    # IMPORTANT:
    # Do not retain the angle after calculating sin/cos.
    # ------------------------------------------------------------------

    doy_zero = doy.astype(np.float32) - np.float32(1.0)

    doy_angle = (
        np.float32(2.0 * np.pi / 365.25)
        * doy_zero
    )

    out["feat_doy_sin"] = np.sin(doy_angle).astype(np.float32)
    out["feat_doy_cos"] = np.cos(doy_angle).astype(np.float32)

    del doy_zero
    del doy_angle

    # ------------------------------------------------------------------
    # Week/month boundaries
    # ------------------------------------------------------------------

    out["feat_is_month_start"] = (dom <= 3).astype(np.int8)
    out["feat_is_month_end"] = (dom >= 28).astype(np.int8)
    out["feat_is_weekend"] = (dow >= 5).astype(np.int8)

    # ------------------------------------------------------------------
    # Holiday features
    # ------------------------------------------------------------------

    holiday_distance = _distance_to_holiday(d)

    out["feat_days_to_holiday"] = holiday_distance
    out["feat_is_holiday"] = (
        holiday_distance == 0
    ).astype(np.int8)

    # Explicitly release local arrays before returning.
    del dow
    del dom
    del month
    del quarter
    del year
    del week
    del doy
    del holiday_distance

    return out


def _distance_to_holiday(d: pd.Series) -> np.ndarray:
    """For each date, return signed offset to nearest major holiday.

    Negative values mean the nearest holiday is in the future.
    Positive values mean the nearest holiday is in the past.
    Zero means the date itself is a holiday.

    The returned array is int16 because the relevant distances are only
    a few hundred days at most.
    """

    # Determine the small holiday lookup table.
    min_year = int(d.dt.year.min())
    max_year = int(d.dt.year.max())

    holiday_days = []

    for year in range(min_year, max_year + 2):
        for month, day in MAJOR_HOLIDAYS:
            holiday_days.append(
                pd.Timestamp(
                    year=year,
                    month=month,
                    day=day,
                )
            )

    holiday_arr = np.array(
        [
            h.value // 86_400_000_000_000
            for h in holiday_days
        ],
        dtype=np.int32,
    )

    # Convert directly to day-resolution integers.
    d_arr = d.to_numpy(
        dtype="datetime64[D]"
    ).astype(np.int32)

    # Locate surrounding holidays.
    idx = np.searchsorted(
        holiday_arr,
        d_arr,
    )

    prev = np.clip(
        idx - 1,
        0,
        len(holiday_arr) - 1,
    )

    nxt = np.clip(
        idx,
        0,
        len(holiday_arr) - 1,
    )

    # Differences are int32 but remain small.
    prev_diff = d_arr - holiday_arr[prev]
    next_diff = holiday_arr[nxt] - d_arr

    # One compact output buffer.
    offset = np.empty(
        len(d_arr),
        dtype=np.int16,
    )

    mask = prev_diff <= next_diff

    offset[mask] = prev_diff[mask]
    offset[~mask] = -next_diff[~mask]

    del d_arr
    del idx
    del prev
    del nxt
    del prev_diff
    del next_diff
    del mask
    del holiday_arr

    return offset
