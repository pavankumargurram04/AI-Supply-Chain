"""Inventory ledger generator.

Given the daily demand series and supplier lead times, simulates a simple
periodic-review (s, Q) policy for every (product, warehouse) pair and emits
inventory movements that are consistent with demand and supplier behaviour.

The output preserves causality:
- Sales reduce on_hand.
- Replenishment orders are placed when inventory position falls below a
  target derived from *actual* local demand, not a fixed constant.
- Stockouts occur exactly when demand exceeds available stock.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from data_platform.generator.config import GeneratorConfig


def _lead_time_days(product_sku: str, supplier_by_sku: Dict[str, Dict[str, Any]]) -> int:
    info = supplier_by_sku.get(product_sku)
    if info is None:
        return 14
    return max(1, int(round(info["lead_time_days"])))


def generate_inventory_flow(
    cfg: GeneratorConfig,
    demand_df: pd.DataFrame,
    supplier_products: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (movements_df, snapshot_df).

    movements: one row per issue / receipt / stockout event.
    snapshot:  final (product, warehouse) on_hand / reserved / inbound.
    """
    # Preferred supplier per product (lowest lead time among preferred).
    supplier_by_sku: Dict[str, Dict[str, Any]] = {}
    for sp in supplier_products:
        cur = supplier_by_sku.get(sp["product_sku"])
        if cur is None or sp["lead_time_days"] < cur["lead_time_days"]:
            supplier_by_sku[sp["product_sku"]] = sp

    movement_rows: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []

    grouped = demand_df.groupby(["product_sku", "warehouse_code"], sort=False)
    for (sku, wh_code), series in grouped:
        series = series.sort_values("demand_date")
        dates = series["demand_date"].to_numpy()
        qty = series["quantity"].to_numpy(dtype=float)
        n = len(dates)

        # Baseline "target stock" ~ 30 days of average demand.
        avg_daily = max(1.0, float(np.mean(qty)))
        target = avg_daily * 30.0
        reorder_qty = max(target * 1.5, avg_daily * 15.0)
        lead = _lead_time_days(sku, supplier_by_sku)

        on_hand = target
        inbound = 0.0
        pending: List[Tuple[int, float]] = []  # (arrival_index, qty)

        # Movement generation.
        for i in range(n):
            d = dates[i]
            # Apply arrivals scheduled for today.
            arrivals = [q for (idx, q) in pending if idx == i]
            if arrivals:
                arrived = float(sum(arrivals))
                on_hand += arrived
                inbound -= arrived
                movement_rows.append({
                    "product_sku": sku,
                    "warehouse_code": wh_code,
                    "movement_type": "receipt",
                    "quantity": arrived,
                    "unit_cost": None,
                    "reference_type": "replenishment",
                    "reference_id": None,
                    "notes": None,
                    "occurred_at": pd.Timestamp(d),
                })
                pending = [(idx, q) for (idx, q) in pending if idx != i]

            demand_today = float(qty[i])

            # Small random adjustment / damage chance.
            if rng.random() < 0.005:
                adj = -float(rng.integers(1, 5))
                on_hand = max(0.0, on_hand + adj)
                movement_rows.append({
                    "product_sku": sku,
                    "warehouse_code": wh_code,
                    "movement_type": "adjustment",
                    "quantity": adj,
                    "unit_cost": None,
                    "reference_type": None,
                    "reference_id": None,
                    "notes": "cycle count adjustment",
                    "occurred_at": pd.Timestamp(d),
                })

            # Issue demand if available, else stockout.
            if demand_today <= on_hand:
                on_hand -= demand_today
                movement_rows.append({
                    "product_sku": sku,
                    "warehouse_code": wh_code,
                    "movement_type": "issue",
                    "quantity": -demand_today,
                    "unit_cost": None,
                    "reference_type": "demand",
                    "reference_id": None,
                    "notes": None,
                    "occurred_at": pd.Timestamp(d),
                })
            else:
                sold = on_hand
                shortfall = demand_today - sold
                if sold > 0:
                    movement_rows.append({
                        "product_sku": sku,
                        "warehouse_code": wh_code,
                        "movement_type": "issue",
                        "quantity": -sold,
                        "unit_cost": None,
                        "reference_type": "demand",
                        "reference_id": None,
                        "notes": "partial fill",
                        "occurred_at": pd.Timestamp(d),
                    })
                movement_rows.append({
                    "product_sku": sku,
                    "warehouse_code": wh_code,
                    "movement_type": "stockout",
                    "quantity": -shortfall,
                    "unit_cost": None,
                    "reference_type": None,
                    "reference_id": None,
                    "notes": "unmet demand",
                    "occurred_at": pd.Timestamp(d),
                })
                on_hand = 0.0

            # Reorder decision — inventory position (on_hand + inbound) below target.
            if on_hand + inbound < target:
                # Place order with stochastic lead time.
                jitter = int(rng.integers(-2, 3))
                arrival = min(n - 1, i + max(1, lead + jitter))
                pending.append((arrival, reorder_qty))
                inbound += reorder_qty

        snapshot_rows.append({
            "product_sku": sku,
            "warehouse_code": wh_code,
            "on_hand": on_hand,
            "reserved": 0.0,
            "inbound": inbound,
            "safety_stock": None,
            "reorder_point": None,
            "last_counted_at": None,
        })

    movements_df = pd.DataFrame(movement_rows)
    snapshot_df = pd.DataFrame(snapshot_rows)
    return movements_df, snapshot_df