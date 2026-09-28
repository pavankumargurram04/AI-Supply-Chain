"""Assemble a DecisionState from curated features + database rows.

The state is a plain dataclass; downstream modules consume fields
individually. The builder never makes predictions — it only gathers inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.logging import get_logger
from data_platform.features.feature_registry import FEATURE_VERSION

log = get_logger(__name__)


@dataclass
class SupplierCandidate:
    supplier_id: Optional[int]
    supplier_code: str
    unit_price: float
    moq: int
    max_order_qty: int
    shipping_cost: float
    lead_time_days: float
    lead_time_std_days: float
    is_preferred: bool
    reliability_class: Optional[str] = None
    risk_probability: float = 0.0
    risk_label: str = "unknown"


@dataclass
class DecisionState:
    product_sku: str
    warehouse_code: str
    service_level: float
    horizon_days: int
    seed: int

    # Demand history (last lookback rows for encoder)
    demand_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Full feature row set for the series (feeds the demand predictor)
    feature_frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Inventory snapshot
    on_hand: float = 0.0
    reserved: float = 0.0
    inbound: float = 0.0

    # Cost parameters derived from product / warehouse
    unit_cost: float = 0.0
    unit_price: float = 0.0
    holding_rate_annual: float = 0.20
    order_fixed_cost: float = 250.0
    shortage_cost_multiplier: float = 3.0

    # Candidates
    suppliers: List[SupplierCandidate] = field(default_factory=list)

    # Supplier-feature rows, one per (supplier_code, product_sku, date) — used
    # to compute supplier risk for each candidate.
    supplier_feature_frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Scenario overrides (already applied by caller)
    demand_multiplier: float = 1.0
    lead_time_override_days: Optional[float] = None
    lead_time_std_override: Optional[float] = None
    inventory_override: Optional[float] = None

    # Versions
    feature_version: str = FEATURE_VERSION
    data_version: Optional[str] = None

    # Snapshot for the trace
    def snapshot(self) -> Dict[str, Any]:
        return {
            "product_sku": self.product_sku,
            "warehouse_code": self.warehouse_code,
            "service_level": self.service_level,
            "horizon_days": self.horizon_days,
            "seed": self.seed,
            "on_hand": self.on_hand,
            "reserved": self.reserved,
            "inbound": self.inbound,
            "unit_cost": self.unit_cost,
            "unit_price": self.unit_price,
            "holding_rate_annual": self.holding_rate_annual,
            "order_fixed_cost": self.order_fixed_cost,
            "shortage_cost_multiplier": self.shortage_cost_multiplier,
            "n_suppliers": len(self.suppliers),
            "supplier_codes": [s.supplier_code for s in self.suppliers],
            "demand_multiplier": self.demand_multiplier,
            "lead_time_override_days": self.lead_time_override_days,
            "lead_time_std_override": self.lead_time_std_override,
            "inventory_override": self.inventory_override,
            "feature_version": self.feature_version,
            "data_version": self.data_version,
        }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_demand_features(features_dir: Path) -> pd.DataFrame:
    candidates = sorted(features_dir.glob("demand_features_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No demand feature files under {features_dir}")
    return pd.read_parquet(candidates[-1])


def _load_supplier_features(features_dir: Path) -> pd.DataFrame:
    candidates = sorted(features_dir.glob("supplier_features_*.parquet"))
    if not candidates:
        return pd.DataFrame()
    return pd.read_parquet(candidates[-1])


def _load_inventory_snapshot(raw_dir: Path, product_sku: str, warehouse_code: str) -> Dict[str, float]:
    """Read the latest inventory snapshot for a (product, warehouse).

    Falls back to zeros if not found; the caller can override via scenario.
    """
    inv_dir = raw_dir / "inventory"
    if inv_dir.is_dir():
        path = inv_dir / "data.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            row = df[(df["product_sku"] == product_sku) & (df["warehouse_code"] == warehouse_code)]
            if not row.empty:
                r = row.iloc[0]
                return {
                    "on_hand": float(r.get("on_hand", 0.0)),
                    "reserved": float(r.get("reserved", 0.0)),
                    "inbound": float(r.get("inbound", 0.0)),
                }
    return {"on_hand": 0.0, "reserved": 0.0, "inbound": 0.0}


def _load_suppliers_for_product(
    raw_dir: Path,
    product_sku: str,
) -> List[SupplierCandidate]:
    """Read supplier_products + suppliers tables from the raw dataset."""
    sp_dir = raw_dir / "supplier_products"
    sup_dir = raw_dir / "suppliers"
    if not sp_dir.exists() or not sup_dir.exists():
        return []

    sp_df = pd.read_parquet(sp_dir / "data.parquet") if (sp_dir / "data.parquet").exists() else pd.read_parquet(sp_dir)
    sup_df = pd.read_parquet(sup_dir / "data.parquet") if (sup_dir / "data.parquet").exists() else pd.read_parquet(sup_dir)

    sp = sp_df[sp_df["product_sku"] == product_sku]
    if sp.empty:
        return []

    merged = sp.merge(sup_df, left_on="supplier_code", right_on="code", how="left", suffixes=("", "_sup"))

    candidates: List[SupplierCandidate] = []
    for _, r in merged.iterrows():
        candidates.append(SupplierCandidate(
            supplier_id=None,
            supplier_code=str(r["supplier_code"]),
            unit_price=float(r["unit_price"]),
            moq=int(r["moq"]),
            max_order_qty=int(r["max_order_qty"]),
            shipping_cost=float(r["shipping_cost"]),
            lead_time_days=float(r["lead_time_days"]),
            lead_time_std_days=float(r["lead_time_std_days"]),
            is_preferred=bool(r.get("is_preferred", False)),
            reliability_class=r.get("reliability_class"),
        ))
    return candidates


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_state(
    *,
    product_sku: str,
    warehouse_code: str,
    curated_features_dir: str | Path,
    raw_dir: Optional[str | Path] = None,
    service_level: float = 0.95,
    horizon_days: int = 30,
    seed: int = 42,
    demand_multiplier: float = 1.0,
    lead_time_override_days: Optional[float] = None,
    lead_time_std_override: Optional[float] = None,
    inventory_override: Optional[float] = None,
) -> DecisionState:
    """Assemble the DecisionState. Fails loudly if required inputs are missing."""
    curated_features_dir = Path(curated_features_dir)
    raw_dir_path = Path(raw_dir) if raw_dir else curated_features_dir.parent.parent / "raw"

    demand_features = _load_demand_features(curated_features_dir)
    mask = (
        (demand_features["product_sku"] == product_sku)
        & (demand_features["warehouse_code"] == warehouse_code)
    )
    series_features = demand_features[mask].sort_values("date").reset_index(drop=True)
    if series_features.empty:
        raise ValueError(
            f"No feature rows for ({product_sku}, {warehouse_code}) in {curated_features_dir}"
        )

    first = series_features.iloc[0]
    state = DecisionState(
        product_sku=product_sku,
        warehouse_code=warehouse_code,
        service_level=float(service_level),
        horizon_days=int(horizon_days),
        seed=int(seed),
        demand_series=series_features[[
            "product_sku", "warehouse_code", "date", "quantity"
        ]].copy(),
        feature_frame=series_features,
        unit_cost=float(first.get("unit_cost", 0.0)),
        unit_price=float(first.get("unit_price", 0.0)),
        demand_multiplier=float(demand_multiplier),
        lead_time_override_days=lead_time_override_days,
        lead_time_std_override=lead_time_std_override,
        inventory_override=inventory_override,
    )

    # Inventory snapshot.
    inv = _load_inventory_snapshot(raw_dir_path, product_sku, warehouse_code)
    state.on_hand = inv["on_hand"]
    state.reserved = inv["reserved"]
    state.inbound = inv["inbound"]
    if inventory_override is not None:
        state.on_hand = float(inventory_override)
        state.reserved = 0.0

    # Suppliers.
    state.suppliers = _load_suppliers_for_product(raw_dir_path, product_sku)
    if not state.suppliers:
        log.warning("decision.state.no_suppliers", extra={
            "product_sku": product_sku, "raw_dir": str(raw_dir_path),
        })

    # Supplier feature frame (used by the risk predictor).
    sup_features = _load_supplier_features(curated_features_dir)
    if not sup_features.empty:
        state.supplier_feature_frame = sup_features[
            sup_features["product_sku"] == product_sku
        ].copy()

    # Apply scenario lead-time overrides to candidates.
    if lead_time_override_days is not None:
        for s in state.suppliers:
            s.lead_time_days = float(lead_time_override_days)
    if lead_time_std_override is not None:
        for s in state.suppliers:
            s.lead_time_std_days = float(lead_time_std_override)

    # Optional data version tag.
    manifest = raw_dir_path / "_manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text())
            state.data_version = payload.get("generated_at")
        except Exception:
            pass

    return state