"""Risk aggregation.

Combines supplier disruption risk with lead-time uncertainty into a single
per-candidate "risk exposure" number. Used by the cost engine to inflate the
expected cost of ordering from risky suppliers.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from decision_engine.state_builder import SupplierCandidate


def risk_multiplier(
    candidate: SupplierCandidate,
    *,
    base_lead_time: float,
    horizon_days: int,
) -> float:
    """A per-candidate multiplier on expected cost.

    The multiplier increases with:
      - the supplier's probability of a disruption within the horizon
      - the ratio of the candidate's lead time to the fastest lead time

    Never returns a value < 1.0.
    """
    # Disruption risk: probability of disruption × a notional loss factor.
    risk_component = 1.0 + 2.0 * candidate.risk_probability

    # Lead-time penalty: (candidate_lt / fastest_lt)^0.5 scaled modestly.
    if base_lead_time > 0:
        lt_ratio = candidate.lead_time_days / base_lead_time
        lt_penalty = 1.0 + 0.10 * math.sqrt(max(lt_ratio, 1.0))
    else:
        lt_penalty = 1.0

    return float(risk_component * lt_penalty)


def exposure(
    candidate: SupplierCandidate,
    *,
    horizon_days: int,
) -> Dict[str, float]:
    """Explicit risk exposure for the trace."""
    return {
        "supplier_code": candidate.supplier_code,
        "disruption_probability": float(candidate.risk_probability),
        "lead_time_mean_days": float(candidate.lead_time_days),
        "lead_time_std_days": float(candidate.lead_time_std_days),
        "horizon_days": int(horizon_days),
    }


def rank_candidates(
    candidates: List[SupplierCandidate],
    *,
    horizon_days: int,
) -> List[Dict[str, float]]:
    """Rank candidates by (unit price × risk multiplier). Deterministic."""
    if not candidates:
        return []
    base_lt = min(c.lead_time_days for c in candidates)
    rows: List[Dict[str, float]] = []
    for c in candidates:
        m = risk_multiplier(c, base_lead_time=base_lt, horizon_days=horizon_days)
        rows.append({
            "supplier_code": c.supplier_code,
            "unit_price": float(c.unit_price),
            "risk_probability": float(c.risk_probability),
            "lead_time_days": float(c.lead_time_days),
            "risk_multiplier": m,
            "risk_adjusted_unit_cost": float(c.unit_price * m),
        })
    rows.sort(key=lambda r: r["risk_adjusted_unit_cost"])
    return rows