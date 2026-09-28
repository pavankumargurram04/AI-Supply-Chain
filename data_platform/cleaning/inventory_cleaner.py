"""Cleaning rules for inventory snapshots and movements."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class InventoryCleanSummary:
    snapshots_in: int
    snapshots_out: int
    movements_in: int
    movements_out: int
    dropped_negative_snapshot: int = 0
    dropped_null_keys: int = 0
    coerced_movement_dates: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def clean_inventory(
    snapshots: pd.DataFrame,
    movements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, InventoryCleanSummary]:
    s_in, m_in = len(snapshots), len(movements)
    summary = InventoryCleanSummary(
        snapshots_in=s_in, snapshots_out=s_in,
        movements_in=m_in, movements_out=m_in,
    )

    if not snapshots.empty:
        snapshots = snapshots.copy()
        before = len(snapshots)
        snapshots = snapshots.dropna(subset=["product_sku", "warehouse_code"])
        summary.dropped_null_keys += before - len(snapshots)

        for c in ("on_hand", "reserved", "inbound"):
            snapshots[c] = pd.to_numeric(snapshots[c], errors="coerce").fillna(0.0)

        neg = (snapshots[["on_hand", "reserved", "inbound"]] < 0).any(axis=1)
        summary.dropped_negative_snapshot = int(neg.sum())
        snapshots = snapshots[~neg]

        # Reserved cannot exceed on_hand.
        over_reserved = snapshots["reserved"] > snapshots["on_hand"]
        snapshots.loc[over_reserved, "reserved"] = snapshots.loc[over_reserved, "on_hand"]

        snapshots = snapshots.reset_index(drop=True)
        summary.snapshots_out = len(snapshots)

    if not movements.empty:
        movements = movements.copy()
        before = len(movements)
        movements["occurred_at"] = pd.to_datetime(movements["occurred_at"], errors="coerce")
        summary.coerced_movement_dates = int(movements["occurred_at"].isna().sum())
        movements = movements.dropna(subset=["occurred_at", "product_sku", "warehouse_code"])
        movements = movements.dropna(subset=["quantity"])
        movements = movements.reset_index(drop=True)
        summary.movements_out = len(movements)

    return snapshots, movements, summary