"""Full end-to-end flow.

Tiny synthetic data → features → demand model (1 epoch) → decision engine
stub → scenario. Verifies that the whole pipeline runs deterministically
and produces consistent numbers.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _build_features(root: Path) -> Path:
    from data_platform.features.feature_registry import FEATURE_VERSION

    rng = np.random.default_rng(0)
    start = date(2022, 1, 1)
    n_days = 90
    rows = []
    for sku in ("SKU-000",):
        for d in range(n_days):
            dt = start + timedelta(days=d)
            q = float(max(0.0, 20.0 + rng.normal(0, 3)))
            rows.append({
                "product_sku": sku, "warehouse_code": "WH-A",
                "date": pd.Timestamp(dt), "quantity": q,
                "feat_demand_lag_1": q, "feat_demand_roll_mean_7": q,
                "feat_demand_roll_std_28": 3.0, "feat_demand_expand_mean": q,
                "feat_promo_count_7": 0, "feat_holiday_count_7": 0,
                "feat_censored_count_7": 0,
                "feat_dow": pd.Timestamp(dt).weekday(),
                "feat_dow_sin": 0.0, "feat_dow_cos": 1.0,
                "feat_dom": dt.day, "feat_month": dt.month,
                "feat_month_sin": 0.0, "feat_month_cos": 1.0,
                "feat_quarter": 1, "feat_year": dt.year,
                "feat_week": dt.isocalendar().week,
                "feat_day_of_year": dt.timetuple().tm_yday,
                "feat_doy_sin": 0.0, "feat_doy_cos": 1.0,
                "feat_is_month_start": 0, "feat_is_month_end": 0,
                "feat_is_weekend": int(pd.Timestamp(dt).weekday() >= 5),
                "feat_is_holiday": 0, "feat_days_to_holiday": 5,
                "category_code": "CAT-A", "unit_cost": 10.0, "unit_price": 25.0,
                "base_demand": 20.0, "lead_time_days": 7,
            })
    fdir = root / "curated" / "e2e" / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        fdir / f"demand_features_{FEATURE_VERSION}.parquet", index=False,
    )

    raw = root / "raw" / "e2e"
    (raw / "supplier_products").mkdir(parents=True, exist_ok=True)
    (raw / "suppliers").mkdir(parents=True, exist_ok=True)
    (raw / "inventory").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "supplier_code": "SUP-A", "product_sku": "SKU-000",
        "unit_price": 10.0, "moq": 10, "max_order_qty": 1000,
        "lead_time_days": 7.0, "lead_time_std_days": 1.5,
        "shipping_cost": 50.0, "is_preferred": True,
    }]).to_parquet(raw / "supplier_products" / "data.parquet", index=False)
    pd.DataFrame([{
        "code": "SUP-A", "name": "A", "reliability_class": "standard",
        "cost_class": "standard", "quality_class": "standard",
        "lead_time_class": "standard", "capacity_class": "medium",
    }]).to_parquet(raw / "suppliers" / "data.parquet", index=False)
    pd.DataFrame([{
        "product_sku": "SKU-000", "warehouse_code": "WH-A",
        "on_hand": 100.0, "reserved": 10.0, "inbound": 0.0,
    }]).to_parquet(raw / "inventory" / "data.parquet", index=False)

    return fdir


def _stub_forecast(state, *, demand_checkpoint_dir, device="cpu"):
    from decision_engine.forecast_decision import ForecastResult
    H = state.horizon_days
    point = np.full(H, 20.0 * state.demand_multiplier, dtype=float)
    std = np.full(H, 3.0, dtype=float)
    origin = state.feature_frame["date"].iloc[-1]
    dates = [(pd.Timestamp(origin) + pd.Timedelta(days=i + 1)).date().isoformat()
             for i in range(H)]
    return ForecastResult(
        point=point, std=std, dates=dates, origin_date=str(origin)[:10],
        model_version="e2e_stub_demand", feature_version=state.feature_version,
    )


def _stub_supplier_risk(state, *, supplier_checkpoint_dir):
    out = []
    for s in state.suppliers:
        s.risk_probability = 0.05
        s.risk_label = "low"
        out.append({
            "supplier_code": s.supplier_code,
            "risk_probability": 0.05, "risk_label": "low",
            "reliability_class": s.reliability_class,
            "lead_time_days": float(s.lead_time_days),
            "lead_time_std_days": float(s.lead_time_std_days),
            "unit_price": float(s.unit_price), "moq": int(s.moq),
            "max_order_qty": int(s.max_order_qty),
            "shipping_cost": float(s.shipping_cost),
            "is_preferred": bool(s.is_preferred),
            "model_version": "e2e_stub_supplier",
        })
    return out


def test_full_flow_reproduces(tmp_path, monkeypatch):
    """Same seed + same inputs → identical final quantity / cost / stockout."""
    from decision_engine import forecast_decision, supplier_decision
    monkeypatch.setattr(forecast_decision, "produce_forecast", _stub_forecast)
    monkeypatch.setattr(supplier_decision, "score_suppliers", _stub_supplier_risk)

    fdir = _build_features(tmp_path)

    from decision_engine.engine import DecisionEngine
    from decision_engine.decision_trace import DecisionRequest
    from scenario_engine import SCENARIOS, run_scenario

    def _run_once():
        engine = DecisionEngine(
            demand_checkpoint_dir=tmp_path / "stub",
            supplier_checkpoint_dir=tmp_path / "stub",
            inventory_checkpoint_dir=None,
            curated_features_dir=fdir,
            raw_dir=tmp_path / "raw" / "e2e",
            simulation_samples=1500,
            device="cpu",
        )
        base = engine.run(DecisionRequest(product_sku="SKU-000", warehouse_code="WH-A", seed=3))
        scen = run_scenario(
            SCENARIOS["demand_spike_50"],
            product_sku="SKU-000", warehouse_code="WH-A",
            engine=engine, seed=3,
        )
        return (
            round(base.final_quantity, 6),
            round(base.total_cost, 6),
            round(base.stockout_probability, 6),
            round(scen.delta["final_quantity"], 6),
        )

    a = _run_once()
    b = _run_once()
    assert a == b


def test_scenario_delta_is_consistent(tmp_path, monkeypatch):
    """Delta equals scenario − baseline for every metric."""
    from decision_engine import forecast_decision, supplier_decision
    monkeypatch.setattr(forecast_decision, "produce_forecast", _stub_forecast)
    monkeypatch.setattr(supplier_decision, "score_suppliers", _stub_supplier_risk)

    fdir = _build_features(tmp_path)

    from decision_engine.engine import DecisionEngine
    from scenario_engine import SCENARIOS, run_scenario

    engine = DecisionEngine(
        demand_checkpoint_dir=tmp_path / "stub",
        supplier_checkpoint_dir=tmp_path / "stub",
        inventory_checkpoint_dir=None,
        curated_features_dir=fdir,
        raw_dir=tmp_path / "raw" / "e2e",
        simulation_samples=1500,
        device="cpu",
    )

    r = run_scenario(SCENARIOS["inventory_5000"], product_sku="SKU-000",
                     warehouse_code="WH-A", engine=engine, seed=3)
    for k, delta in r.delta.items():
        expected = r.scenario[k] - r.baseline[k]
        assert abs(delta - expected) < 1e-6, f"delta mismatch for {k}"