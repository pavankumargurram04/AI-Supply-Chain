"""Inventory-related features.

Static per (product, warehouse) features: current on-hand, reserved,
inbound, position, coverage. These become the "state" that downstream
inventory decision logic reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_inventory_features(inventory: pd.DataFrame) -> pd.DataFrame:
    """Derive coverage and derived ratios from raw snapshot columns."""
    if inventory.empty:
        return inventory

    out = inventory.copy()
    out["on_hand"] = out["on_hand"].astype("float32")
    out["reserved"] = out["reserved"].astype("float32")
    out["inbound"] = out["inbound"].astype("float32")

    out["feat_inv_available"] = (out["on_hand"] - out["reserved"]).astype("float32")
    out["feat_inv_position"] = (out["on_hand"] - out["reserved"] + out["inbound"]).astype("float32")
    out["feat_inv_reserved_ratio"] = (
        out["reserved"] / np.maximum(out["on_hand"], 1.0)
    ).astype("float32")
    out["feat_inv_inbound_ratio"] = (
        out["inbound"] / np.maximum(out["on_hand"], 1.0)
    ).astype("float32")
    return out