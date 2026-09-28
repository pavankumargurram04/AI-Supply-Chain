"""Demand time-series generator.

For every (product, warehouse) pair, produces a daily demand series that
combines:

  base level × warehouse multiplier × category multiplier
  + yearly seasonality (amplitude depends on seasonality_class)
  + linear trend (sign depends on lifecycle_stage)
  + weekly pattern
  + monthly pattern
  + holiday flags (with demand impact)
  + promotional days
  + rare random shocks
  + gaussian noise (scale depends on volatility_class)

Returns a pandas DataFrame with natural keys (sku, warehouse_code, date).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.enums import LifecycleStage, SeasonalityClass, VolatilityClass
from data_platform.generator.config import GeneratorConfig


# ---------------- Lookup tables ----------------
_SEASONALITY_AMPLITUDE = {
    SeasonalityClass.NONE.value: 0.00,
    SeasonalityClass.LOW.value: 0.08,
    SeasonalityClass.MEDIUM.value: 0.18,
    SeasonalityClass.HIGH.value: 0.32,
    SeasonalityClass.VERY_HIGH.value: 0.50,
}

_VOLATILITY_NOISE = {
    VolatilityClass.LOW.value: 0.10,
    VolatilityClass.MEDIUM.value: 0.25,
    VolatilityClass.HIGH.value: 0.50,
    VolatilityClass.EXTREME.value: 0.90,
}

# Trend per day (roughly year-over-year percent scaled by days).
_LIFECYCLE_TREND = {
    LifecycleStage.INTRODUCTION.value: 0.0009,
    LifecycleStage.GROWTH.value: 0.0006,
    LifecycleStage.MATURITY.value: 0.0000,
    LifecycleStage.DECLINE.value: -0.0007,
    LifecycleStage.DISCONTINUED.value: -0.0020,
}

_WEEKDAY_MULT = np.array([1.00, 1.02, 1.02, 1.02, 1.05, 1.15, 0.80])  # Mon..Sun
_MONTH_MULT = np.array([0.94, 0.92, 1.00, 1.00, 1.02, 1.03, 1.00, 1.00, 1.05, 1.05, 1.14, 1.18])

_HOLIDAY_DATES = [
    (1, 1),   # New Year
    (5, 1),   # Labour Day
    (7, 4),   # US Independence
    (11, 25), # Black Friday window start (approx)
    (12, 24),
    (12, 25),
    (12, 31),
]
_HOLIDAY_MULT = {
    (1, 1): 0.35,
    (5, 1): 0.70,
    (7, 4): 0.85,
    (11, 25): 1.55,
    (12, 24): 1.30,
    (12, 25): 0.20,
    (12, 31): 1.10,
}


def _holiday_multiplier(month: int, day: int) -> float:
    return _HOLIDAY_MULT.get((month, day), 1.0)


# ---------------- Core per-series generator ----------------
def _generate_series(
    *,
    sku: str,
    warehouse_code: str,
    dates: pd.DatetimeIndex,
    base_demand: float,
    seasonality_class: str,
    volatility_class: str,
    lifecycle_stage: str,
    warehouse_mult: float,
    category_mult: float,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(dates)
    t = np.arange(n, dtype=np.float64)
    weekday = dates.weekday.values
    month = dates.month.values
    day = dates.day.values

    # Base level.
    level = base_demand * warehouse_mult * category_mult

    # Yearly seasonality — random phase per series.
    amp = _SEASONALITY_AMPLITUDE.get(seasonality_class, 0.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    seas = 1.0 + amp * np.sin(2.0 * np.pi * t / 365.25 + phase)

    # Trend.
    trend_rate = _LIFECYCLE_TREND.get(lifecycle_stage, 0.0)
    trend = 1.0 + trend_rate * t

    # Calendar effects.
    wk = _WEEKDAY_MULT[weekday]
    mo = _MONTH_MULT[month - 1]
    hol = np.array([_holiday_multiplier(int(m), int(d)) for m, d in zip(month, day)])

    # Promotions — Poisson arrival, lasting 1–5 days.
    promo = np.zeros(n, dtype=bool)
    if cfg.enable_promotion:
        # ~30 promotional days per year per series.
        rate = 30.0 / 365.25
        arrivals = rng.random(n) < rate
        idx = np.where(arrivals)[0]
        for i in idx:
            length = int(rng.integers(1, 6))
            promo[i : i + length] = True
    promo_mult = np.where(promo, rng.uniform(1.4, 2.4), 1.0)

    # Rare shocks — multiplicative, lasting 3–10 days.
    shock = np.ones(n, dtype=np.float64)
    if cfg.enable_shocks:
        shock_rate = 3.0 / 365.25
        shock_arrivals = rng.random(n) < shock_rate
        idx = np.where(shock_arrivals)[0]
        for i in idx:
            length = int(rng.integers(3, 11))
            magnitude = float(rng.choice([0.35, 0.55, 1.6, 2.2, 3.0], p=[0.15, 0.15, 0.3, 0.25, 0.15]))
            shock[i : i + length] *= magnitude

    # Noise — Gaussian, always positive after clip.
    noise_sigma = _VOLATILITY_NOISE.get(volatility_class, 0.25)

    mean_series = level * seas * trend * wk * mo * hol * promo_mult * shock
    noise = rng.normal(loc=1.0, scale=noise_sigma, size=n)
    raw = mean_series * noise
    raw = np.clip(raw, 0.0, None)
    quantity = np.round(raw, 4)

    # Holiday flags (a holiday actually occurs on that date).
    holiday_flag = np.array(
        [(_holiday_multiplier(int(m), int(d)) != 1.0) for m, d in zip(month, day)],
        dtype=bool,
    )

    return pd.DataFrame({
        "product_sku": sku,
        "warehouse_code": warehouse_code,
        "demand_date": dates.date,
        "quantity": quantity,
        "promotion_flag": promo,
        "holiday_flag": holiday_flag,
        "is_censored": False,  # set later by inventory flow when demand exceeds stock
        "source": "synthetic",
    })


# ---------------- Public entry point ----------------
def generate_demand(
    cfg: GeneratorConfig,
    products: List[Dict[str, Any]],
    warehouses: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate demand for every (product, warehouse) pair across the whole date range."""
    dates = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="D")

    # Warehouse demand multipliers — larger DCs sell more.
    wh_mult: Dict[str, float] = {}
    for wh in warehouses:
        base = 1.0
        if wh["region"] == "AMER":
            base = 1.15
        elif wh["region"] == "EMEA":
            base = 1.00
        elif wh["region"] == "APAC":
            base = 1.10
        scale = float(np.sqrt(wh["capacity_units"] / 500_000.0))
        wh_mult[wh["code"]] = float(np.clip(base * scale, 0.4, 2.5))

    frames: List[pd.DataFrame] = []
    for product in products:
        cat_code = product["category_code"].split("-")[0]
        # Simple category multiplier — some categories sell more units per SKU.
        cat_mult = {
            "BEVG": 1.4, "GROC": 1.5, "FOOD": 1.3, "STAT": 1.3,
            "ELEC": 0.9, "APPL": 0.7, "FURN": 0.5, "AUTO": 0.6,
        }.get(cat_code, 1.0)

        for wh in warehouses:
            df = _generate_series(
                sku=product["sku"],
                warehouse_code=wh["code"],
                dates=dates,
                base_demand=float(product["base_demand"]),
                seasonality_class=product["seasonality_class"],
                volatility_class=product["volatility_class"],
                lifecycle_stage=product["lifecycle_stage"],
                warehouse_mult=wh_mult[wh["code"]],
                category_mult=cat_mult,
                cfg=cfg,
                rng=rng,
            )
            frames.append(df)

    return pd.concat(frames, ignore_index=True)