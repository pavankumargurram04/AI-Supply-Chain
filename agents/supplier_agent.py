"""Supplier agent.

Wraps `decision_engine.supplier_decision.score_suppliers`. Emits a payload
with the per-supplier risk probability, label, lead time, and commercial
terms. Every numeric field comes from the risk model or the DB.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agents.base_agent import BaseAgent
from core.logging import get_logger
from decision_engine.supplier_decision import score_suppliers

log = get_logger(__name__)


class SupplierAgent(BaseAgent):
    name = "supplier"

    def run(self, context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
        state = context["state"]
        checkpoint_dir = context["supplier_checkpoint_dir"]

        rows = score_suppliers(state, supplier_checkpoint_dir=checkpoint_dir)
        payload = {"suppliers": rows, "n_suppliers": len(rows)}

        model_versions: Dict[str, str] = {}
        if rows and rows[0].get("model_version"):
            model_versions["supplier_model"] = rows[0]["model_version"]

        warnings: List[str] = []
        if not rows:
            warnings.append("no supplier candidates found for this product")
        high_risk = [r for r in rows if r.get("risk_probability", 0.0) >= 0.5]
        if high_risk:
            warnings.append(
                f"{len(high_risk)} of {len(rows)} candidates have risk ≥ 0.5"
            )
        return payload, model_versions, warnings