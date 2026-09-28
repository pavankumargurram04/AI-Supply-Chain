"""Duplicate-key detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd


@dataclass
class DuplicateReport:
    table: str
    key_columns: Sequence[str]
    duplicate_rows: int
    distinct_duplicate_keys: int
    examples: List[dict]

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "key_columns": list(self.key_columns),
            "duplicate_rows": self.duplicate_rows,
            "distinct_duplicate_keys": self.distinct_duplicate_keys,
            "examples": self.examples,
        }


def detect_duplicates(
    df: pd.DataFrame,
    table_name: str,
    key_columns: Sequence[str],
    *,
    example_limit: int = 5,
) -> DuplicateReport:
    cols = [c for c in key_columns if c in df.columns]
    if not cols or df.empty:
        return DuplicateReport(table_name, key_columns, 0, 0, [])

    dup_mask = df.duplicated(subset=cols, keep=False)
    dup_df = df[dup_mask]
    distinct_keys = dup_df[cols].drop_duplicates().shape[0]
    examples = dup_df[cols].head(example_limit).to_dict(orient="records")

    return DuplicateReport(
        table=table_name,
        key_columns=cols,
        duplicate_rows=int(dup_mask.sum()),
        distinct_duplicate_keys=int(distinct_keys),
        examples=examples,
    )


# ---------------------------------------------------------------------------
# Canonical keys per table (matching the DB uniques).
# ---------------------------------------------------------------------------

KEYS: Dict[str, List[str]] = {
    "categories": ["code"],
    "products": ["sku"],
    "suppliers": ["code"],
    "warehouses": ["code"],
    "supplier_products": ["supplier_code", "product_sku"],
    "demand_records": ["product_sku", "warehouse_code", "demand_date"],
    "inventory": ["product_sku", "warehouse_code"],
    "supplier_events": [],       # events are append-only; no natural unique
    "inventory_movements": [],   # append-only ledger
}