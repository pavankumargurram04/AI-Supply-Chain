"""Hard constraints that no model output may violate.

These are the last line of defence before a recommendation is persisted.
They are checked *after* the optimizer runs. A violation is not a
threshold on the business outcome — it is a structural sanity check
(e.g. "don't recommend a negative quantity", "don't exceed budget").

Guardrails are all-or-nothing: if any are violated, the recommendation is
returned with `guardrail_violations` populated and callers decide whether
to approve, reject, or escalate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GuardrailResult:
    violations: List[str]

    @property
    def passed(self) -> bool:
        return not self.violations


def check_guardrails(
    *,
    recommended_quantity: float,
    allocation_total_cost: float,
    budget: Optional[float],
    service_level_target: float,
    predicted_service_level: float,
    lead_time_days: float,
) -> GuardrailResult:
    """Return all structural violations, if any."""
    v: List[str] = []

    if recommended_quantity < 0:
        v.append(f"recommended_quantity ({recommended_quantity:.2f}) is negative")

    if allocation_total_cost < 0:
        v.append(f"allocation_total_cost ({allocation_total_cost:.2f}) is negative")

    if budget is not None and allocation_total_cost > budget + 1e-6:
        v.append(
            f"allocation cost {allocation_total_cost:.2f} exceeds budget {budget:.2f}"
        )

    if lead_time_days <= 0:
        v.append(f"lead_time_days ({lead_time_days}) must be positive")

    # Service level sanity: it must be a probability, not an error metric.
    if not (0.0 <= predicted_service_level <= 1.0):
        v.append(
            f"predicted_service_level ({predicted_service_level}) not in [0, 1]"
        )

    if not (0.5 <= service_level_target <= 0.999):
        v.append(
            f"service_level_target ({service_level_target}) outside supported range [0.5, 0.999]"
        )

    return GuardrailResult(violations=v)