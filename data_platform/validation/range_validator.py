"""Range / domain validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class RangeRule:
    column: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    allowed_values: Optional[set] = None
    forbid_null: bool = True


@dataclass
class RangeIssue:
    table: str
    column: str
    kind: str          # "below_min" | "above_max" | "not_allowed" | "null"
    offending_count: int
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "column": self.column,
            "kind": self.kind,
            "offending_count": self.offending_count,
            "detail": self.detail,
        }


def validate_ranges(df: pd.DataFrame, table_name: str, rules: List[RangeRule]) -> List[RangeIssue]:
    issues: List[RangeIssue] = []
    n = len(df)

    for rule in rules:
        if rule.column not in df.columns:
            continue
        col = df[rule.column]

        if rule.forbid_null:
            nulls = int(col.isna().sum())
            if nulls > 0:
                issues.append(RangeIssue(table_name, rule.column, "null", nulls, f"{nulls}/{n}"))

        if rule.minimum is not None:
            below = int((col < rule.minimum).sum())
            if below > 0:
                issues.append(
                    RangeIssue(table_name, rule.column, "below_min", below, f"min={rule.minimum}")
                )

        if rule.maximum is not None:
            above = int((col > rule.maximum).sum())
            if above > 0:
                issues.append(
                    RangeIssue(table_name, rule.column, "above_max", above, f"max={rule.maximum}")
                )

        if rule.allowed_values is not None:
            bad = int((~col.isin(rule.allowed_values) & col.notna()).sum())
            if bad > 0:
                issues.append(
                    RangeIssue(
                        table_name, rule.column, "not_allowed", bad,
                        f"allowed={sorted(rule.allowed_values)[:5]}",
                    )
                )

    return issues


# ---------------------------------------------------------------------------
# Canonical rule sets
# ---------------------------------------------------------------------------

RULES: Dict[str, List[RangeRule]] = {
    "products": [
        RangeRule("unit_cost", minimum=0.0),
        RangeRule("unit_price", minimum=0.0),
        RangeRule("base_demand", minimum=0.0),
        RangeRule("lead_time_days", minimum=1, maximum=365),
        RangeRule(
            "lifecycle_stage",
            allowed_values={"introduction", "growth", "maturity", "decline", "discontinued"},
        ),
        RangeRule(
            "volatility_class",
            allowed_values={"low", "medium", "high", "extreme"},
        ),
    ],
    "suppliers": [
        RangeRule("contracted_capacity_units", minimum=0.0),
        RangeRule(
            "reliability_class",
            allowed_values={"premium", "reliable", "standard", "poor"},
        ),
    ],
    "warehouses": [
        RangeRule("capacity_units", minimum=0.0),
        RangeRule("operating_cost_daily", minimum=0.0),
    ],
    "supplier_products": [
        RangeRule("unit_price", minimum=0.0),
        RangeRule("moq", minimum=1),
        RangeRule("max_order_qty", minimum=1),
        RangeRule("lead_time_days", minimum=0.0),
        RangeRule("lead_time_std_days", minimum=0.0),
        RangeRule("shipping_cost", minimum=0.0),
    ],
    "demand_records": [
        RangeRule("quantity", minimum=0.0),
    ],
    "inventory": [
        RangeRule("on_hand", minimum=0.0),
        RangeRule("reserved", minimum=0.0),
        RangeRule("inbound", minimum=0.0),
    ],
    "inventory_movements": [
        RangeRule(
            "movement_type",
            allowed_values={
                "receipt", "issue", "transfer_in", "transfer_out",
                "adjustment", "return", "damage",
                "reservation", "reservation_release", "stockout",
            },
        ),
    ],
    "supplier_events": [
        RangeRule(
            "event_type",
            allowed_values={
                "delay", "early", "quality_incident", "capacity_outage",
                "price_change", "fill_rate_drop", "on_time_miss", "recovery",
            },
        ),
        RangeRule("severity", allowed_values={"low", "medium", "high"}),
    ],
}