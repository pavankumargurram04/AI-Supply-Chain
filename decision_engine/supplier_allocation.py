"""Bridge between the decision engine and ml.inventory.allocation.

Converts DecisionState candidates into SupplierOffer objects and invokes the
greedy risk-adjusted allocator. If anything fails, returns a degraded result
with an explanatory note rather than silently using a fallback rule.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from core.logging import get_logger
from decision_engine.state_builder import DecisionState
from ml.inventory.allocation import SupplierOffer, allocate_suppliers

log = get_logger(__name__)


def allocate(
    state: DecisionState,
    *,
    required_quantity: float,
    budget: Optional[float] = None,
    max_suppliers: Optional[int] = None,
    risk_weight: float = 2.0,
    preferred_supplier_codes: Optional[List[str]] = None,
) -> Dict:
    if not state.suppliers:
        return {
            "requested_quantity": float(required_quantity),
            "allocated_quantity": 0.0,
            "unallocated_quantity": float(required_quantity),
            "total_purchase_cost": 0.0,
            "total_shipping_cost": 0.0,
            "total_cost": 0.0,
            "weighted_lead_time": 0.0,
            "weighted_risk": 0.0,
            "n_suppliers": 0,
            "lines": [],
            "constraints": {"quantity_satisfied": required_quantity <= 0},
            "notes": ["no supplier candidates available"],
        }

    offers: List[SupplierOffer] = []
    for s in state.suppliers:
        offers.append(SupplierOffer(
            supplier_id=s.supplier_id if s.supplier_id is not None else 0,
            supplier_code=s.supplier_code,
            unit_price=float(s.unit_price),
            moq=int(s.moq),
            max_order_qty=int(s.max_order_qty),
            shipping_cost=float(s.shipping_cost),
            lead_time_days=float(s.lead_time_days),
            risk_probability=float(s.risk_probability),
        ))

    preferred_ids: Optional[List[int]] = None
    if preferred_supplier_codes:
        pref_set = set(preferred_supplier_codes)
        preferred_ids = [
            (s.supplier_id if s.supplier_id is not None else i)
            for i, s in enumerate(state.suppliers)
            if s.supplier_code in pref_set
        ]

    result = allocate_suppliers(
        required_quantity=float(required_quantity),
        offers=offers,
        budget=budget,
        max_suppliers=max_suppliers,
        risk_weight=risk_weight,
        preferred_supplier_ids=preferred_ids,
    )
    return result.to_dict()