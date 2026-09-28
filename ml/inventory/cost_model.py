"""Cost model primitives.

All cost rates are *parameters* — the values come from the database, the
product, the warehouse, or explicit scenario input. This module does not
contain any default rate that is used as a hidden decision rule.

Three cost components:
    holding:  h * average_inventory * time_years
    ordering: K * n_orders + shipping_cost * n_orders
    shortage: p * lost_units

Where:
    h = holding_rate * unit_cost       (annual fraction of unit cost)
    K = fixed order cost per order      (from configuration / product)
    p = shortage_cost_per_unit          (from configuration / product)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CostParameters:
    """Every rate is explicit; no hidden defaults.

    Values are provided by the caller. `to_dict` makes them JSON-serialisable
    for provenance.
    """
    unit_cost: float                       # per unit, from product / supplier
    holding_rate_annual: float             # fraction of unit_cost held per year
    order_fixed_cost: float                # per purchase order
    shortage_cost_per_unit: float          # per unit of unmet demand
    shipping_cost_per_order: float = 0.0   # from supplier
    review_period_days: int = 1            # review cadence (for scaling)
    days_per_year: int = 365

    # Derived rate used by cost functions.
    @property
    def holding_rate_daily(self) -> float:
        return self.holding_rate_annual / float(self.days_per_year)

    def to_dict(self) -> dict:
        return {
            "unit_cost": self.unit_cost,
            "holding_rate_annual": self.holding_rate_annual,
            "order_fixed_cost": self.order_fixed_cost,
            "shortage_cost_per_unit": self.shortage_cost_per_unit,
            "shipping_cost_per_order": self.shipping_cost_per_order,
            "review_period_days": self.review_period_days,
            "days_per_year": self.days_per_year,
        }


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def holding_cost(
    average_inventory_units: float,
    params: CostParameters,
    *,
    days: int = 1,
) -> float:
    """Holding cost for one review period.

    average_inventory_units is the mean on-hand over the period.
    """
    return (
        average_inventory_units
        * params.unit_cost
        * params.holding_rate_daily
        * float(days)
    )


def ordering_cost(
    n_orders: int,
    params: CostParameters,
) -> float:
    """Fixed cost of placing `n_orders` orders plus shipping per order."""
    per_order = params.order_fixed_cost + params.shipping_cost_per_order
    return float(n_orders) * per_order


def shortage_cost(
    unmet_units: float,
    params: CostParameters,
) -> float:
    """Cost of unsatisfied demand."""
    return max(0.0, float(unmet_units)) * params.shortage_cost_per_unit


def total_cost(
    *,
    average_inventory_units: float,
    n_orders: int,
    unmet_units: float,
    params: CostParameters,
    days: int = 1,
) -> dict:
    """Return component-wise and total cost for a period."""
    h = holding_cost(average_inventory_units, params, days=days)
    o = ordering_cost(n_orders, params)
    s = shortage_cost(unmet_units, params)
    return {
        "holding": h,
        "ordering": o,
        "shortage": s,
        "total": h + o + s,
        "days": days,
        "params": params.to_dict(),
    }