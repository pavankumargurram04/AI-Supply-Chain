"""Reorder point derivation.

Reorder point = expected lead-time demand + safety stock.

Or, when simulation is available and authoritative, the direct α-quantile of
the lead-time-demand distribution. Both approaches are exposed; the decision
engine stores which one it used.
"""

from __future__ import annotations

from typing import Optional

from ml.inventory.safety_stock import safety_stock_from_simulation
from ml.inventory.simulation import LeadTimeDemandResult


def reorder_point_from_simulation(
    ltd_result: LeadTimeDemandResult,
    *,
    service_level: float,
    method: str = "sum",
) -> float:
    """Return reorder point (units).

    method="sum"    : E[LTD] + safety_stock_from_simulation
    method="direct" : α-quantile of LTD (equivalent, but computed directly)
    """
    if method == "sum":
        ss = safety_stock_from_simulation(ltd_result, service_level=service_level)
        return float(ltd_result.mean + ss)
    if method == "direct":
        q = ltd_result.alpha_quantile
        # If alpha differs from config, interpolate.
        if service_level in ltd_result.quantiles:
            return float(ltd_result.quantiles[service_level])
        import numpy as np
        return float(np.interp(
            service_level,
            np.asarray(ltd_result.sample_quantiles),
            np.asarray(ltd_result.sample_values),
        ))
    raise ValueError(f"Unknown method: {method}")


def reorder_point_normal(
    *,
    mean_daily_demand: float,
    lead_time_mean: float,
    safety_stock: float,
) -> float:
    """Expected lead-time demand + safety stock (analytical fallback)."""
    return float(mean_daily_demand * lead_time_mean + safety_stock)