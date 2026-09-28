"""Total landed cost per candidate.

For a candidate supplier and a chosen quantity Q:
    landed_unit = unit_price × risk_multiplier
    total = Q × landed_unit + shipping_cost
    shortage_expected = Q_shortage × unit_cost × shortage_multiplier

The cost engine produces numbers for the trace. It does not decide anything.
"""

from __future__ import annotations

from typing import Any, Dict

from decision_engine.risk_engine import risk_multiplier
from decision_engine.state_builder import DecisionState, SupplierCandidate


def landed_cost(
    candidate: SupplierCandidate,
    *,
    quantity: float,
    state: DecisionState,
    base_lead_time: float,
) -> Dict[str, Any]:
    """Return a landed-cost breakdown for one candidate + quantity."""
    mult = risk_multiplier(
        candidate,
        base_lead_time=base_lead_time,
        horizon_days=state.horizon_days,
    )
    landed_unit = candidate.unit_price * mult
    purchase = landed_unit * quantity
    shipping = candidate.shipping_cost
    total = purchase + shipping

    # Alternative view: pure purchase cost without risk inflation, for audit.
    purchase_nominal = candidate.unit_price * quantity

    return {
        "supplier_code": candidate.supplier_code,
        "quantity": float(quantity),
        "unit_price": float(candidate.unit_price),
        "risk_multiplier": float(mult),
        "landed_unit": float(landed_unit),
        "purchase_nominal": float(purchase_nominal),
        "purchase_risk_adjusted": float(purchase),
        "shipping": float(shipping),
        "total": float(total),
    }