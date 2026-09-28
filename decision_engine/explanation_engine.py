"""Explain the recommendation using ONLY numeric evidence from the trace.

Never fabricate prose. Every sentence is a template filled by a number drawn
from the trace.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}"


def build_explanation(trace_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return {"explanation": str, "evidence": [...]}.

    `trace_dict` is the DecisionTrace as a dict.
    """
    evidence: List[Dict[str, Any]] = []

    def add(kind: str, text: str, value: Any) -> None:
        evidence.append({"kind": kind, "text": text, "value": value})

    parts: List[str] = []

    # 1) Forecast summary
    fc = trace_dict.get("forecast_point", [])
    if fc:
        total_fc = float(sum(fc))
        mean_fc = total_fc / max(1, len(fc))
        parts.append(
            f"Forecast horizon covers {len(fc)} days; mean daily demand "
            f"{_fmt(mean_fc)} units, total expected demand {_fmt(total_fc)} units."
        )
        add("forecast", "total expected demand over horizon", total_fc)

    # 2) Inventory status
    inv = trace_dict.get("inventory_state", {})
    available = float(inv.get("available", 0.0))
    inv_pos = float(inv.get("inventory_position", 0.0))
    parts.append(
        f"Current available inventory is {_fmt(available)} units "
        f"(position {_fmt(inv_pos)} units including inbound)."
    )
    add("inventory", "available inventory", available)
    add("inventory", "inventory position", inv_pos)

    # 3) Lead-time demand and stockout
    ltd = trace_dict.get("ltd_simulation", {})
    mean_ltd = float(ltd.get("mean", 0.0))
    p_stockout = float(trace_dict.get("stockout_probability", 0.0))
    parts.append(
        f"Expected lead-time demand is {_fmt(mean_ltd)} units; "
        f"estimated stockout probability {p_stockout:.2%}."
    )
    add("ltd", "expected lead-time demand", mean_ltd)
    add("stockout", "P(stockout during lead time)", p_stockout)

    # 4) Safety stock and ROP
    ss = float(trace_dict.get("safety_stock", 0.0))
    rop = float(trace_dict.get("reorder_point", 0.0))
    parts.append(
        f"To hold service level at {trace_dict['request'].get('service_level'):.2f}, "
        f"safety stock is {_fmt(ss)} units and reorder point is {_fmt(rop)} units."
    )
    add("safety_stock", "safety stock", ss)
    add("reorder_point", "reorder point", rop)

    # 5) DQN
    dqn = trace_dict.get("dqn_action")
    if dqn:
        parts.append(
            f"The inventory policy suggested an order multiplier of "
            f"{_fmt(float(dqn.get('multiplier', 0.0)), 2)}× expected lead-time demand "
            f"(≈ {_fmt(float(dqn.get('order_quantity', 0.0)))} units)."
        )
        add("dqn", "policy order quantity", dqn.get("order_quantity"))

    # 6) Final quantity + allocation
    final_qty = float(trace_dict.get("final_quantity", 0.0))
    n_suppliers = len(trace_dict.get("final_suppliers", []))
    cost = float(trace_dict.get("total_cost", 0.0))
    parts.append(
        f"Recommended order quantity: {_fmt(final_qty)} units split across "
        f"{n_suppliers} supplier(s); expected total landed cost {_fmt(cost)}."
    )
    add("final_qty", "recommended order quantity", final_qty)
    add("final_cost", "expected total landed cost", cost)

    # 7) Supplier selection rationale
    for sup in trace_dict.get("final_suppliers", []):
        code = sup.get("supplier_code", "?")
        qty = float(sup.get("quantity", 0.0))
        risk = float(sup.get("risk_probability", 0.0))
        lt = float(sup.get("lead_time_days", 0.0))
        parts.append(
            f"  • {code}: {_fmt(qty)} units, risk {risk:.2%}, "
            f"lead time {_fmt(lt, 1)} days."
        )
        add("supplier", f"{code} quantity", qty)
        add("supplier", f"{code} risk", risk)
        add("supplier", f"{code} lead time", lt)

    # 8) Service level
    sl = float(trace_dict.get("expected_service_level", 0.0))
    parts.append(
        f"Expected service level under the recommended plan is {sl:.2%} "
        f"(target {trace_dict['request'].get('service_level'):.2%})."
    )
    add("service_level", "expected service level", sl)

    # 9) Warnings
    for w in trace_dict.get("warnings", []):
        parts.append(f"Warning: {w}")

    return {
        "explanation": "\n".join(parts),
        "evidence": evidence,
    }