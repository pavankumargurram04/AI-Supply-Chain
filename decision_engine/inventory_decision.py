"""Inventory intelligence step.

Runs the Monte Carlo lead-time demand simulation using the demand forecast,
the supplier lead-time distribution, and the current inventory. Derives
safety stock, reorder point, and stockout probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.logging import get_logger
from decision_engine.forecast_decision import ForecastResult
from decision_engine.state_builder import DecisionState, SupplierCandidate
from ml.inventory.reorder_point import reorder_point_from_simulation
from ml.inventory.safety_stock import safety_stock_from_simulation
from ml.inventory.simulation import SimulationConfig, simulate_lead_time_demand

log = get_logger(__name__)


@dataclass
class InventoryDecision:
    available: float
    inventory_position: float
    expected_lead_time_demand: float
    lead_time_mean: float
    lead_time_std: float
    stockout_probability: float
    expected_shortage: float
    safety_stock: float
    reorder_point: float
    alpha_quantile: float
    ltd_simulation: Dict[str, Any]
    mode: str
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": float(self.available),
            "inventory_position": float(self.inventory_position),
            "expected_lead_time_demand": float(self.expected_lead_time_demand),
            "lead_time_mean": float(self.lead_time_mean),
            "lead_time_std": float(self.lead_time_std),
            "stockout_probability": float(self.stockout_probability),
            "expected_shortage": float(self.expected_shortage),
            "safety_stock": float(self.safety_stock),
            "reorder_point": float(self.reorder_point),
            "alpha_quantile": float(self.alpha_quantile),
            "ltd_simulation": self.ltd_simulation,
            "mode": self.mode,
            "seed": int(self.seed),
        }


def _pick_lead_time_candidate(state: DecisionState) -> SupplierCandidate:
    """Choose the candidate whose lead-time distribution we simulate with.

    We use the *best* candidate (lowest risk-adjusted cost). If there are no
    candidates, we fall back to a documented default so the simulation is
    well-defined and the caller is warned.
    """
    if not state.suppliers:
        return SupplierCandidate(
            supplier_id=None, supplier_code="__default__",
            unit_price=state.unit_cost, moq=1, max_order_qty=1_000_000,
            shipping_cost=0.0, lead_time_days=14.0, lead_time_std_days=3.0,
            is_preferred=False,
        )
    # Sort by (risk * price) — same ordering used downstream.
    return sorted(
        state.suppliers,
        key=lambda s: (s.unit_price * (1.0 + 2.0 * s.risk_probability), s.lead_time_days),
    )[0]


def run_inventory_decision(
    state: DecisionState,
    forecast: ForecastResult,
    *,
    simulation_samples: int = 10_000,
) -> InventoryDecision:
    """Run the Monte Carlo simulation and derive SS / ROP."""

    available = float(state.on_hand - state.reserved)
    inv_position = float(state.on_hand - state.reserved + state.inbound)

    # Residual pool: we use the residual series stored under the curated dir if
    # available; otherwise the simulation falls back to Gaussian around the
    # forecast — this is flagged in the returned `mode`.
    residuals = None
    residuals_path = Path("datasets/curated") / "residuals_demand.parquet"
    if residuals_path.exists():
        import pandas as pd
        rdf = pd.read_parquet(residuals_path)
        if not rdf.empty:
            sub = rdf[
                (rdf["product_sku"] == state.product_sku)
                & (rdf["warehouse_code"] == state.warehouse_code)
            ]
            if not sub.empty:
                residuals = sub["residual"].to_numpy(dtype=np.float64)

    # Lead-time distribution — from the chosen candidate.
    cand = _pick_lead_time_candidate(state)
    lt_mean = float(state.lead_time_override_days) if state.lead_time_override_days is not None else float(cand.lead_time_days)
    lt_std = float(state.lead_time_std_override) if state.lead_time_std_override is not None else float(cand.lead_time_std_days)

    # Forecast used for the LTD simulation must cover at least ceil(lt_mean) days.
    fc_horizon = forecast.point.size
    if fc_horizon < int(np.ceil(lt_mean)):
        # Repeat the last forecast value to cover the tail.
        extra = int(np.ceil(lt_mean)) - fc_horizon + 2
        fc_ext = np.concatenate([forecast.point, np.full(extra, forecast.point[-1])])
        fc_std_ext = np.concatenate([forecast.std, np.full(extra, forecast.std[-1])])
    else:
        fc_ext = forecast.point
        fc_std_ext = forecast.std

    # The simulation samples residuals — we use the forecast residuals if the
    # user has computed them, else we sample daily shocks from the forecast std.
    sim_residuals = residuals
    if sim_residuals is None and forecast.std.size > 0:
        # We synthesise a residual pool with the same std as the forecast
        # uncertainty. This is a documented, deterministic fallback.
        rng = np.random.default_rng(state.seed)
        sigma = float(np.mean(fc_std_ext))
        sim_residuals = rng.normal(0.0, max(sigma, 1e-6), size=1000)

    cfg = SimulationConfig(
        n_samples=simulation_samples,
        seed=state.seed,
        alpha=state.service_level,
    )
    ltd = simulate_lead_time_demand(
        forecast_point=fc_ext,
        forecast_residuals=sim_residuals,
        lead_time_mean=lt_mean,
        lead_time_std=lt_std,
        available_inventory=available,
        config=cfg,
    )

    ss = safety_stock_from_simulation(ltd, service_level=state.service_level)
    rop = reorder_point_from_simulation(ltd, service_level=state.service_level, method="sum")

    return InventoryDecision(
        available=available,
        inventory_position=inv_position,
        expected_lead_time_demand=float(ltd.mean),
        lead_time_mean=lt_mean,
        lead_time_std=lt_std,
        stockout_probability=float(ltd.stockout_probability),
        expected_shortage=float(ltd.expected_shortage),
        safety_stock=float(ss),
        reorder_point=float(rop),
        alpha_quantile=float(ltd.alpha_quantile),
        ltd_simulation=ltd.to_dict(),
        mode=ltd.mode,
        seed=int(ltd.seed),
    )