"""Cleaning rules for supplier performance events.

Ensures event types and severities are canonical and timestamps are
parseable. Events referencing unknown suppliers are dropped (referential
integrity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

import pandas as pd


_CANONICAL_EVENT_TYPES = {
    "delay", "early", "quality_incident", "capacity_outage",
    "price_change", "fill_rate_drop", "on_time_miss", "recovery",
}
_CANONICAL_SEVERITIES = {"low", "medium", "high"}


@dataclass
class SupplierCleanSummary:
    rows_in: int
    rows_out: int
    dropped_null_supplier: int = 0
    dropped_unknown_supplier: int = 0
    coerced_timestamps: int = 0
    normalised_event_types: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def clean_supplier_events(
    df: pd.DataFrame,
    known_supplier_codes: Iterable[str],
) -> tuple[pd.DataFrame, SupplierCleanSummary]:
    summary = SupplierCleanSummary(rows_in=len(df), rows_out=0)
    if df.empty:
        return df, summary

    df = df.copy()
    known = set(known_supplier_codes)

    before = len(df)
    df = df.dropna(subset=["supplier_code"])
    summary.dropped_null_supplier = before - len(df)

    before = len(df)
    df = df[df["supplier_code"].isin(known)]
    summary.dropped_unknown_supplier = before - len(df)

    before = len(df)
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], errors="coerce")
    df = df.dropna(subset=["occurred_at"])
    summary.coerced_timestamps = before - len(df)

    # Normalise event type strings.
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    bad_type = ~df["event_type"].isin(_CANONICAL_EVENT_TYPES)
    summary.normalised_event_types = int(bad_type.sum())
    df = df[~bad_type]

    df["severity"] = df["severity"].astype(str).str.lower()
    df.loc[~df["severity"].isin(_CANONICAL_SEVERITIES), "severity"] = "medium"

    summary.rows_out = len(df)
    return df.reset_index(drop=True), summary