"""Supplier allocation for a procurement requirement.

Given:
    - required quantity Q
    - candidate suppliers with unit price, MOQ, max quantity, shipping, risk, lead time

Produce:
    - an allocation (list of {supplier_id, quantity, unit_price, line_total})
    - total cost (purchase + shipping)
    - expected lead-time-weighted risk
    - constraint satisfaction flags

Method: greedy sorted by "expected risk-adjusted cost per unit":
    score = unit_price + risk_penalty(risk) + shipping/n * 1e-3
where risk_penalty scales the per-unit cost by a risk factor that the
caller controls (default 2.0). This is deterministic and easy to audit;
later we can swap in a full MILP without changing the interface.

The function respects:
    - MOQ (a supplier is either ordered ≥ MOQ or skipped)
    - max_order_qty
    - budget (if provided)
    - max_suppliers (optional cap)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SupplierOffer:
    supplier_id: int
    supplier_code: str
    unit_price: float
    moq: int
    max_order_qty: int
    shipping_cost: float
    lead_time_days: float
    risk_probability: float


@dataclass
class AllocationLine:
    supplier_id: int
    supplier_code: str
    quantity: float
    unit_price: float
    line_total: float
    shipping_cost: float
    lead_time_days: float
    risk_probability: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "supplier_id": self.supplier_id,
            "supplier_code": self.supplier_code,
            "quantity": float(self.quantity),
            "unit_price": float(self.unit_price),
            "line_total": float(self.line_total),
            "shipping_cost": float(self.shipping_cost),
            "lead_time_days": float(self.lead_time_days),
            "risk_probability": float(self.risk_probability),
            "rationale": self.rationale,
        }


@dataclass
class AllocationResult:
    requested_quantity: float
    allocated_quantity: float
    unallocated_quantity: float
    total_purchase_cost: float
    total_shipping_cost: float
    total_cost: float
    weighted_lead_time: float
    weighted_risk: float
    n_suppliers: int
    lines: List[AllocationLine] = field(default_factory=list)
    constraints: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requested_quantity": float(self.requested_quantity),
            "allocated_quantity": float(self.allocated_quantity),
            "unallocated_quantity": float(self.unallocated_quantity),
            "total_purchase_cost": float(self.total_purchase_cost),
            "total_shipping_cost": float(self.total_shipping_cost),
            "total_cost": float(self.total_cost),
            "weighted_lead_time": float(self.weighted_lead_time),
            "weighted_risk": float(self.weighted_risk),
            "n_suppliers": int(self.n_suppliers),
            "lines": [l.to_dict() for l in self.lines],
            "constraints": dict(self.constraints),
            "notes": list(self.notes),
        }


def _score(offer: SupplierOffer, risk_weight: float) -> float:
    """Lower is better."""
    risk_pen = risk_weight * offer.risk_probability * offer.unit_price
    # Shipping is amortised: small per-unit cost, but non-zero so ties break
    # towards cheaper shipping.
    return offer.unit_price + risk_pen + 1e-4 * offer.shipping_cost


def allocate_suppliers(
    *,
    required_quantity: float,
    offers: List[SupplierOffer],
    budget: Optional[float] = None,
    max_suppliers: Optional[int] = None,
    risk_weight: float = 2.0,
    preferred_supplier_ids: Optional[List[int]] = None,
) -> AllocationResult:
    """Greedy risk-adjusted allocation. Deterministic given inputs."""
    if required_quantity <= 0:
        return AllocationResult(
            requested_quantity=float(required_quantity),
            allocated_quantity=0.0,
            unallocated_quantity=float(required_quantity),
            total_purchase_cost=0.0,
            total_shipping_cost=0.0,
            total_cost=0.0,
            weighted_lead_time=0.0,
            weighted_risk=0.0,
            n_suppliers=0,
            lines=[],
            constraints={"quantity_satisfied": True, "budget_respected": True},
            notes=["zero required quantity"],
        )

    # Sort by score (lower is better).
    preferred = set(preferred_supplier_ids or [])
    sorted_offers = sorted(
        offers,
        key=lambda o: (
            0 if o.supplier_id in preferred else 1,  # preferred first
            _score(o, risk_weight),
            o.lead_time_days,
        ),
    )

    lines: List[AllocationLine] = []
    remaining = float(required_quantity)
    total_purchase = 0.0
    total_shipping = 0.0
    budget_remaining = float(budget) if budget is not None else float("inf")

    for offer in sorted_offers:
        if remaining <= 0:
            break
        if max_suppliers is not None and len(lines) >= max_suppliers:
            break

        # Skip suppliers whose MOQ is above the remaining need
        if offer.moq > remaining:
            continue

        # Take as much as we can (respecting max_order_qty).
        # Leave a feasible remainder for other suppliers under their MOQ constraints.
        take = min(remaining, offer.max_order_qty)

        remaining_offers = [
            other for other in sorted_offers if other.supplier_id != offer.supplier_id
        ]

        if remaining > take and remaining_offers:
            max_future_capacity = sum(
                min(other.max_order_qty, remaining)
                for other in remaining_offers
                if other.moq <= remaining
            )
            min_future_moq = min(
                (other.moq for other in remaining_offers if other.moq <= remaining),
                default=0,
            )

            if remaining - take < min_future_moq:
                feasible_take = remaining - min(max_future_capacity, remaining - min_future_moq)
                if feasible_take >= offer.moq:
                    take = min(take, feasible_take)

        if take < offer.moq:
            continue

        line_cost = take * offer.unit_price
        if line_cost + offer.shipping_cost > budget_remaining:
            # Try to fit a smaller quantity if budget-constrained.
            headroom = budget_remaining - offer.shipping_cost
            if headroom < offer.moq * offer.unit_price:
                continue
            take = min(take, headroom / offer.unit_price)
            take = float(int(take))  # integer units
            if take < offer.moq:
                continue
            line_cost = take * offer.unit_price

        lines.append(
            AllocationLine(
                supplier_id=offer.supplier_id,
                supplier_code=offer.supplier_code,
                quantity=float(take),
                unit_price=float(offer.unit_price),
                line_total=float(line_cost),
                shipping_cost=float(offer.shipping_cost),
                lead_time_days=float(offer.lead_time_days),
                risk_probability=float(offer.risk_probability),
                rationale=(
                    f"selected by risk-adjusted cost score "
                    f"(score={_score(offer, risk_weight):.4f}, "
                    f"risk={offer.risk_probability:.3f}, lt={offer.lead_time_days:.1f}d)"
                ),
            )
        )
        total_purchase += line_cost
        total_shipping += offer.shipping_cost
        budget_remaining -= line_cost + offer.shipping_cost
        remaining -= take

    allocated = float(sum(l.quantity for l in lines))
    unallocated = float(required_quantity - allocated)
    total_cost = total_purchase + total_shipping

    weighted_lt = (
        sum(l.quantity * l.lead_time_days for l in lines) / allocated
        if allocated > 0
        else 0.0
    )
    weighted_risk = (
        sum(l.quantity * l.risk_probability for l in lines) / allocated
        if allocated > 0
        else 0.0
    )

    constraints = {
        "quantity_satisfied": unallocated <= 1e-6,
        "budget_respected": budget is None or total_cost <= float(budget) + 1e-6,
        "max_suppliers_respected": max_suppliers is None or len(lines) <= max_suppliers,
    }
    notes: List[str] = []
    if not constraints["quantity_satisfied"]:
        notes.append(f"only {allocated:.2f}/{required_quantity:.2f} units allocated")
    if not constraints["budget_respected"]:
        notes.append("budget exceeded")
    if not lines:
        notes.append("no supplier met MOQ / capacity constraints")

    return AllocationResult(
        requested_quantity=float(required_quantity),
        allocated_quantity=allocated,
        unallocated_quantity=unallocated,
        total_purchase_cost=float(total_purchase),
        total_shipping_cost=float(total_shipping),
        total_cost=float(total_cost),
        weighted_lead_time=float(weighted_lt),
        weighted_risk=float(weighted_risk),
        n_suppliers=len(lines),
        lines=lines,
        constraints=constraints,
        notes=notes,
    )