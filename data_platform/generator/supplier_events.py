"""Supplier performance event generator.

For every supplier, produces realistic performance events (delays,
quality incidents, capacity outages, price changes, fill-rate drops)
whose rates depend on the supplier's reliability / quality classes.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List

import numpy as np

from data_platform.generator.config import GeneratorConfig


# Monthly event arrival rates, keyed by class.
_DELAY_RATE_PER_MONTH = {
    "premium": 0.05,
    "reliable": 0.20,
    "standard": 0.55,
    "poor": 1.40,
}
_QUALITY_RATE_PER_MONTH = {
    "high": 0.08,
    "standard": 0.25,
    "low": 0.75,
}
_OUTAGE_RATE_PER_MONTH = {
    "small": 0.30,
    "medium": 0.15,
    "large": 0.05,
}
_PRICE_CHANGE_RATE_PER_MONTH = 0.10


def _random_datetime(rng: np.random.Generator, start: date, end: date) -> datetime:
    total_days = (end - start).days
    if total_days <= 0:
        return datetime.combine(start, time(12, 0))
    d = start + timedelta(days=int(rng.integers(0, total_days)))
    return datetime.combine(d, time(int(rng.integers(0, 24)), int(rng.integers(0, 60))))


def generate_supplier_events(
    cfg: GeneratorConfig,
    suppliers: List[Dict[str, Any]],
    products: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    months = cfg.days / 30.0
    skus = [p["sku"] for p in products]

    rows: List[Dict[str, Any]] = []
    for supplier in suppliers:
        reliability = supplier["reliability_class"]
        quality = supplier["quality_class"]
        capacity = supplier["capacity_class"]

        n_delay = int(rng.poisson(_DELAY_RATE_PER_MONTH[reliability] * months))
        n_quality = int(rng.poisson(_QUALITY_RATE_PER_MONTH[quality] * months))
        n_outage = int(rng.poisson(_OUTAGE_RATE_PER_MONTH[capacity] * months))
        n_price = int(rng.poisson(_PRICE_CHANGE_RATE_PER_MONTH * months))

        for _ in range(n_delay):
            rows.append({
                "supplier_code": supplier["code"],
                "product_sku": str(rng.choice(skus)) if rng.random() < 0.5 else None,
                "event_type": "delay",
                "severity": str(rng.choice(["low", "medium", "high"], p=[0.4, 0.4, 0.2])),
                "payload": {"delay_days": int(rng.integers(1, 15))},
                "description": "Shipment delayed",
                "occurred_at": _random_datetime(rng, cfg.start_date, cfg.end_date),
            })

        for _ in range(n_quality):
            rows.append({
                "supplier_code": supplier["code"],
                "product_sku": str(rng.choice(skus)) if rng.random() < 0.7 else None,
                "event_type": "quality_incident",
                "severity": str(rng.choice(["low", "medium", "high"], p=[0.3, 0.5, 0.2])),
                "payload": {"units_affected": int(rng.integers(1, 500))},
                "description": "Quality incident",
                "occurred_at": _random_datetime(rng, cfg.start_date, cfg.end_date),
            })

        for _ in range(n_outage):
            rows.append({
                "supplier_code": supplier["code"],
                "product_sku": None,
                "event_type": "capacity_outage",
                "severity": "high",
                "payload": {"duration_days": int(rng.integers(2, 20))},
                "description": "Capacity outage",
                "occurred_at": _random_datetime(rng, cfg.start_date, cfg.end_date),
            })

        for _ in range(n_price):
            rows.append({
                "supplier_code": supplier["code"],
                "product_sku": None,
                "event_type": "price_change",
                "severity": "low",
                "payload": {"pct_change": float(rng.uniform(-0.10, 0.20))},
                "description": "Contracted price adjusted",
                "occurred_at": _random_datetime(rng, cfg.start_date, cfg.end_date),
            })

    rows.sort(key=lambda r: r["occurred_at"])
    return rows