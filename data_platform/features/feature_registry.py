"""Feature registry — declarative list of every feature the platform uses.

Each feature has:
- name
- kind (temporal / demand / supplier / inventory / static)
- dtype family
- description
- source (which generator / pipeline produces it)
- lag-safe (bool) — False means the feature is not safe as a *predictor*

FEATURE_VERSION is content-addressed from the registry contents so that any
change (add / remove / rename / reorder) produces a new version string.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    dtype: str
    description: str
    source: str
    lag_safe: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Demand features
# ---------------------------------------------------------------------------
_DEMAND: List[FeatureSpec] = []

for lag in (1, 2, 3, 7, 14, 21, 28, 56, 91, 182, 365):
    _DEMAND.append(FeatureSpec(
        name=f"feat_demand_lag_{lag}", kind="demand", dtype="float32",
        description=f"Demand quantity lagged {lag} days (per series)",
        source="add_demand_lags",
    ))

for w in (7, 14, 28, 56, 91, 180, 365):
    for stat in ("mean", "std", "max", "min"):
        _DEMAND.append(FeatureSpec(
            name=f"feat_demand_roll_{stat}_{w}", kind="demand", dtype="float32",
            description=f"Rolling {stat} of demand over last {w} days (lagged)",
            source="add_demand_rolling",
        ))

for stat in ("mean", "std"):
    _DEMAND.append(FeatureSpec(
        name=f"feat_demand_expand_{stat}", kind="demand", dtype="float32",
        description=f"Expanding {stat} of demand up to t-1",
        source="add_demand_expanding",
    ))

for w in (7, 28):
    _DEMAND.append(FeatureSpec(
        name=f"feat_promo_count_{w}", kind="demand", dtype="int8",
        description=f"Number of promotion days in last {w} days (lagged)",
        source="add_promotion_history",
    ))
    _DEMAND.append(FeatureSpec(
        name=f"feat_holiday_count_{w}", kind="demand", dtype="int8",
        description=f"Number of holiday days in last {w} days (lagged)",
        source="add_promotion_history",
    ))
    _DEMAND.append(FeatureSpec(
        name=f"feat_censored_count_{w}", kind="demand", dtype="int8",
        description=f"Number of censored (stockout) days in last {w} days (lagged)",
        source="add_censored_features",
    ))


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------
_TEMPORAL: List[FeatureSpec] = [
    FeatureSpec("feat_dow",           "temporal", "int16",  "Day of week (0=Mon)",     "add_calendar_features"),
    FeatureSpec("feat_dom",           "temporal", "int8",   "Day of month",            "add_calendar_features"),
    FeatureSpec("feat_month",         "temporal", "int8",   "Month (1–12)",            "add_calendar_features"),
    FeatureSpec("feat_quarter",       "temporal", "int8",   "Quarter (1–4)",           "add_calendar_features"),
    FeatureSpec("feat_year",          "temporal", "int16",  "Year",                    "add_calendar_features"),
    FeatureSpec("feat_week",          "temporal", "int16",  "ISO week",                "add_calendar_features"),
    FeatureSpec("feat_day_of_year",   "temporal", "int16",  "Day of year",             "add_calendar_features"),
    FeatureSpec("feat_dow_sin",       "temporal", "float32","Cyclical dow sine",       "add_calendar_features"),
    FeatureSpec("feat_dow_cos",       "temporal", "float32","Cyclical dow cosine",     "add_calendar_features"),
    FeatureSpec("feat_month_sin",     "temporal", "float32","Cyclical month sine",     "add_calendar_features"),
    FeatureSpec("feat_month_cos",     "temporal", "float32","Cyclical month cosine",   "add_calendar_features"),
    FeatureSpec("feat_doy_sin",       "temporal", "float32","Cyclical doy sine",       "add_calendar_features"),
    FeatureSpec("feat_doy_cos",       "temporal", "float32","Cyclical doy cosine",     "add_calendar_features"),
    FeatureSpec("feat_is_month_start","temporal", "int8",   "First 3 days of month",   "add_calendar_features"),
    FeatureSpec("feat_is_month_end",  "temporal", "int8",   "Last 3 days of month",    "add_calendar_features"),
    FeatureSpec("feat_is_weekend",    "temporal", "int8",   "Sat/Sun flag",            "add_calendar_features"),
    FeatureSpec("feat_days_to_holiday","temporal","int16",  "Signed offset to nearest major holiday","add_calendar_features"),
    FeatureSpec("feat_is_holiday",    "temporal", "int8",   "Holiday flag",            "add_calendar_features"),
]


# ---------------------------------------------------------------------------
# Supplier features
# ---------------------------------------------------------------------------
_SUPPLIER: List[FeatureSpec] = []

for w in (30, 90, 180, 365):
    for col in ("delay_count", "quality_count", "outage_count",
                "price_change_count", "fill_drop_count", "delay_days_total"):
        _SUPPLIER.append(FeatureSpec(
            name=f"feat_sup_{col}_{w}", kind="supplier", dtype="float32",
            description=f"Supplier {col} rolling sum over {w} days (lagged)",
            source="add_supplier_features",
        ))
    _SUPPLIER.append(FeatureSpec(
        name=f"feat_sup_ontime_proxy_{w}", kind="supplier", dtype="float32",
        description=f"On-time proxy over {w} days (lagged)",
        source="add_supplier_features",
    ))

for kind in ("delay", "quality", "outage"):
    _SUPPLIER.append(FeatureSpec(
        name=f"feat_sup_days_since_{kind}", kind="supplier", dtype="float32",
        description=f"Days since most recent {kind} event (lagged)",
        source="add_supplier_features",
    ))

for static in ("reliability", "cost_class", "quality_class",
               "lead_time_class", "capacity_class"):
    _SUPPLIER.append(FeatureSpec(
        name=f"feat_sup_{static}", kind="supplier", dtype="category",
        description=f"Supplier static attribute: {static}",
        source="add_supplier_features",
    ))

for col in ("lead_time_days", "lead_time_std_days", "unit_price",
            "moq", "shipping_cost"):
    _SUPPLIER.append(FeatureSpec(
        name=f"feat_sup_{col}", kind="supplier", dtype="float32",
        description=f"Supplier-product contract term: {col}",
        source="add_supplier_features",
    ))


# ---------------------------------------------------------------------------
# Inventory features
# ---------------------------------------------------------------------------
_INVENTORY: List[FeatureSpec] = [
    FeatureSpec("feat_inv_available",       "inventory", "float32", "On hand - reserved",          "add_inventory_features"),
    FeatureSpec("feat_inv_position",        "inventory", "float32", "On hand - reserved + inbound","add_inventory_features"),
    FeatureSpec("feat_inv_reserved_ratio",  "inventory", "float32", "Reserved / on_hand",          "add_inventory_features"),
    FeatureSpec("feat_inv_inbound_ratio",   "inventory", "float32", "Inbound / on_hand",           "add_inventory_features"),
]


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
ALL_FEATURES: List[FeatureSpec] = _TEMPORAL + _DEMAND + _SUPPLIER + _INVENTORY
REGISTRY: Dict[str, FeatureSpec] = {f.name: f for f in ALL_FEATURES}


def _compute_version() -> str:
    """Content-address the registry so any change produces a new version."""
    payload = json.dumps(
        [f.to_dict() for f in ALL_FEATURES], sort_keys=True, separators=(",", ":")
    )
    return "v1_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


FEATURE_VERSION: str = _compute_version()


def feature_columns(kind: str | None = None) -> List[str]:
    if kind is None:
        return list(REGISTRY.keys())
    return [name for name, spec in REGISTRY.items() if spec.kind == kind]