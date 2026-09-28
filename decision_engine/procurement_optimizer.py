"""Procurement optimizer.

Determines *how much* to order (using the DQN policy if available, else the
reorder-point net requirement), then delegates the *who* to the allocation
module.

The DQN policy's output quantity is treated as a *candidate* alongside the
reorder-point-derived requirement. Both candidates are evaluated numerically,
and the smaller "sufficient" one is chosen to preserve cost discipline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.logging import get_logger
from decision_engine.inventory_decision import InventoryDecision
from decision_engine.state_builder import DecisionState
from decision_engine.supplier_allocation import allocate

log = get_logger(__name__)


def _dqn_candidate_quantity(
    state: DecisionState,
    *,
    inventory_checkpoint_dir: Optional[str | Path],
    expected_ltd: float,
) -> Optional[Dict[str, Any]]:
    """If a DQN checkpoint is available, evaluate the greedy policy at the
    current state and return the order quantity as a *candidate*."""
    if inventory_checkpoint_dir is None:
        return None
    ckpt = Path(inventory_checkpoint_dir)
    if not ckpt.exists():
        return None
    try:
        from ml.inventory.policy import InventoryPolicy

        policy = InventoryPolicy(ckpt, device="cpu")

        # Build a single-step state vector matching the DQN training format.
        params = policy.env_cfg["cost"]
        available = float(state.on_hand - state.reserved)
        inbound = float(state.inbound)
        fc_mean = float(expected_ltd / max(1.0, state.horizon_days))
        fc_std = float(0.20 * fc_mean)
        lt_mean = float(state.suppliers[0].lead_time_days) if state.suppliers else 7.0
        lt_std = float(state.suppliers[0].lead_time_std_days) if state.suppliers else 1.5
        risk = float(state.suppliers[0].risk_probability) if state.suppliers else 0.1

        # Normalisers must match EnvConfig defaults.
        inv_norm = 500.0
        forecast_norm = 100.0
        lead_time_norm = 30.0

        svec = np.array([
            (available) / inv_norm,
            inbound / inv_norm,
            fc_mean / forecast_norm,
            fc_std / forecast_norm,
            lt_mean / lead_time_norm,
            lt_std / lead_time_norm,
            risk,
            state.service_level,
            params["holding_rate_annual"],
            params["shortage_cost_per_unit"] / max(params["unit_cost"], 1e-6),
            0.5,
        ], dtype=np.float32)

        decision = policy.decide_from_state(svec, expected_lead_time_demand=expected_ltd)
        return decision.to_dict()
    except Exception as e:
        log.warning("decision.optimizer.dqn_failed", extra={"error": str(e)})
        return None


def optimize(
    state: DecisionState,
    inventory: InventoryDecision,
    *,
    budget: Optional[float] = None,
    max_suppliers: Optional[int] = None,
    preferred_supplier_codes: Optional[List[str]] = None,
    inventory_checkpoint_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Return the final recommendation dict.

    Candidates considered:
        - dqn_qty:    the DQN's greedy output quantity (if available)
        - rop_net:    the reorder-point net requirement (max(0, rop − position))

    The chosen quantity is the *smallest candidate that closes the stockout
    gap*, computed as the smallest candidate ≥ rop_net. If none reach rop_net,
    the largest candidate is chosen.
    """
    dqn = _dqn_candidate_quantity(
        state, inventory_checkpoint_dir=inventory_checkpoint_dir,
        expected_ltd=inventory.expected_lead_time_demand,
    )

    inv_position = float(inventory.inventory_position)
    rop = float(inventory.reorder_point)
    rop_net = max(0.0, rop - inv_position)

    candidates: List[Dict[str, Any]] = []
    if dqn is not None:
        candidates.append({
            "source": "dqn",
            "quantity": float(dqn["order_quantity"]),
            "details": dqn,
        })
    candidates.append({
        "source": "reorder_point_net",
        "quantity": float(rop_net),
        "details": {"inventory_position": inv_position, "reorder_point": rop},
    })

    # Chooser: smallest candidate ≥ rop_net, else largest.
    above = [c for c in candidates if c["quantity"] >= rop_net]
    if above:
        chosen = min(above, key=lambda c: c["quantity"])
    else:
        chosen = max(candidates, key=lambda c: c["quantity"])

    final_qty = float(chosen["quantity"])

    # Allocation
    alloc = allocate(
        state,
        required_quantity=final_qty,
        budget=budget,
        max_suppliers=max_suppliers,
        preferred_supplier_codes=preferred_supplier_codes,
    )

    return {
        "candidates": candidates,
        "chosen_source": chosen["source"],
        "final_quantity": final_qty,
        "allocation": alloc,
        "expected_service_level": 1.0 - float(inventory.stockout_probability),
    }