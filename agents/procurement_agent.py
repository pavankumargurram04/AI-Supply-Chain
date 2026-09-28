"""Procurement agent.

Wraps `decision_engine.procurement_optimizer.optimize` and the supplier
allocation bridge. Produces the final quantity, allocation, cost, and DQN
action (if available).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agents.base_agent import BaseAgent
from core.logging import get_logger
from decision_engine.inventory_decision import InventoryDecision
from decision_engine.procurement_optimizer import optimize

log = get_logger(__name__)


class ProcurementAgent(BaseAgent):
    name = "procurement"

    def run(self, context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
        state = context["state"]
        inventory_msg = context.get("inventory_message")
        if inventory_msg is None or inventory_msg.status == "error":
            raise RuntimeError("Procurement agent requires a successful inventory message.")

        # Rebuild the InventoryDecision from the inventory agent payload.
        inv = inventory_msg.payload
        inventory_decision = InventoryDecision(
            available=float(inv["available"]),
            inventory_position=float(inv["inventory_position"]),
            expected_lead_time_demand=float(inv["expected_lead_time_demand"]),
            lead_time_mean=float(inv["lead_time_mean"]),
            lead_time_std=float(inv["lead_time_std"]),
            stockout_probability=float(inv["stockout_probability"]),
            expected_shortage=float(inv["expected_shortage"]),
            safety_stock=float(inv["safety_stock"]),
            reorder_point=float(inv["reorder_point"]),
            alpha_quantile=float(inv["alpha_quantile"]),
            ltd_simulation=dict(inv["ltd_simulation"]),
            mode=str(inv["mode"]),
            seed=int(inv["seed"]),
        )

        request = context["request"]
        opt = optimize(
            state,
            inventory_decision,
            budget=request.budget,
            max_suppliers=request.max_suppliers,
            preferred_supplier_codes=request.preferred_supplier_codes,
            inventory_checkpoint_dir=context.get("inventory_checkpoint_dir"),
        )

        # Collect model versions from the DQN candidate if used.
        model_versions: Dict[str, str] = {}
        for c in opt["candidates"]:
            if c["source"] == "dqn":
                details = c.get("details") or {}
                if details.get("model_version"):
                    model_versions["inventory_model"] = details["model_version"]

        warnings: List[str] = []
        alloc = opt["allocation"]
        if not alloc.get("constraints", {}).get("quantity_satisfied", True):
            warnings.append("allocation could not satisfy the full requested quantity")
        if not alloc.get("lines"):
            warnings.append("no supplier allocation produced")

        return opt, model_versions, warnings