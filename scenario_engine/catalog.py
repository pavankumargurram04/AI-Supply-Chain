"""Predefined scenarios.

Each `ScenarioSpec` is a pure description — no numbers are invented, only
overrides applied to the request. The decision engine produces all outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioSpec:
    key: str
    scenario_type: str
    label: str
    description: str

    # Overrides applied to DecisionRequest (None = leave as-is).
    demand_multiplier: Optional[float] = None
    lead_time_override_days: Optional[float] = None
    lead_time_std_override: Optional[float] = None
    inventory_override: Optional[float] = None
    budget: Optional[float] = None
    max_suppliers: Optional[int] = None
    service_level: Optional[float] = None

    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _spec(**kw: Any) -> ScenarioSpec:
    return ScenarioSpec(**kw)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
SCENARIOS: Dict[str, ScenarioSpec] = {}


def _register(spec: ScenarioSpec) -> None:
    if spec.key in SCENARIOS:
        raise ValueError(f"Duplicate scenario key: {spec.key}")
    SCENARIOS[spec.key] = spec


# Demand spikes.
for pct in (10, 20, 50, 100):
    _register(_spec(
        key=f"demand_spike_{pct}",
        scenario_type="demand_spike",
        label=f"Demand +{pct}%",
        description=f"Baseline demand multiplier of {1 + pct/100:.2f}×.",
        demand_multiplier=1.0 + pct / 100.0,
        tags=["demand"],
    ))

# Lead-time increases.
for days in (1, 3, 7, 14):
    _register(_spec(
        key=f"lead_time_plus_{days}",
        scenario_type="lead_time_increase",
        label=f"Lead time +{days}d",
        description=f"All candidates' lead time fixed at baseline + {days} days.",
        lead_time_override_days=None,  # computed at run time from baseline
        tags=["lead_time"],
        # Store delta for the runner:
    ))
    SCENARIOS[f"lead_time_plus_{days}"].__dict__["_lead_time_delta"] = days

# Supplier failure.
_register(_spec(
    key="supplier_failure",
    scenario_type="supplier_failure",
    label="Single supplier failure",
    description="All candidates' risk set to 1.0 and lead time inflated by 30 days.",
    lead_time_override_days=None,
    tags=["supplier"],
))

# Capacity reduction.
_register(_spec(
    key="capacity_reduction_50",
    scenario_type="supplier_capacity_reduction",
    label="Capacity −50%",
    description="Every candidate's max_order_qty halved at run time.",
    tags=["supplier"],
))

# Inventory override levels.
for level in (50, 100, 500, 1000, 5000, 10000):
    _register(_spec(
        key=f"inventory_{level}",
        scenario_type="inventory_level",
        label=f"Inventory = {level}",
        description=f"Force on-hand inventory to {level} units.",
        inventory_override=float(level),
        tags=["inventory"],
    ))

# Budget constraint.
_register(_spec(
    key="budget_constraint",
    scenario_type="budget_constraint",
    label="Budget-limited order",
    description="Budget cap applied to the allocation stage.",
    budget=5000.0,
    tags=["procurement"],
))

# Service target.
_register(_spec(
    key="service_target_99",
    scenario_type="service_target",
    label="Service target 99%",
    description="Raise target service level to 0.99.",
    service_level=0.99,
    tags=["service"],
))
_register(_spec(
    key="service_target_90",
    scenario_type="service_target",
    label="Service target 90%",
    description="Lower target service level to 0.90.",
    service_level=0.90,
    tags=["service"],
))


def get_scenario(key: str) -> ScenarioSpec:
    if key not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario {key!r}. Available: {sorted(SCENARIOS)[:10]}…"
        )
    return SCENARIOS[key]


def list_scenarios(tag: Optional[str] = None) -> List[ScenarioSpec]:
    specs = list(SCENARIOS.values())
    if tag is not None:
        specs = [s for s in specs if tag in s.tags]
    return sorted(specs, key=lambda s: s.key)