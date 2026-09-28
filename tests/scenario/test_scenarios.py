"""Scenario engine tests using stubbed models.

We monkeypatch produce_forecast and score_suppliers so the scenario engine
can be exercised without trained checkpoints.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _build_artifacts(root: Path) -> tuple[Path, Path]:
    """Write tiny curated features + raw master tables under root."""
    from data_platform.features.feature_registry import FEATURE_VERSION

    start = date(2022, 1, 1)
    n_days = 120
    rng = np.random.default_rng(0)

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

    fdir = root / "curated" / "test" / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        fdir / f"demand_features_{FEATURE_VERSION}.parquet", index=False,
    )

    sup_rows = []
    for sc in ("SUP-A", "SUP-B"):
        for d in range(n_days):
            dt = start + timedelta(days=d)
            sup_rows.append({
                "supplier_code": sc, "product_sku": "SKU-000",
                "date": pd.Timestamp(dt),
                "feat_sup_delay_count_30": 0.0, "feat_sup_delay_count_90": 0.0,
                "feat_sup_quality_count_30": 0.0, "feat_sup_outage_count_90": 0.0,
                "feat_sup_ontime_proxy_30": 0.9, "feat_sup_days_since_delay": 10.0,
                "feat_sup_lead_time_days": 7.0, "feat_sup_lead_time_std_days": 1.5,
                "feat_sup_unit_price": 10.0, "feat_sup_moq": 10.0,
                "feat_sup_shipping_cost": 50.0,
                "feat_sup_reliability": "standard",
                "feat_sup_cost_class": "standard",
                "feat_sup_quality_class": "standard",
                "feat_sup_lead_time_class": "standard",
                "feat_sup_capacity_class": "medium",
            })
    pd.DataFrame(sup_rows).to_parquet(
        fdir / f"supplier_features_{FEATURE_VERSION}.parquet", index=False,
    )

    raw = root / "raw" / "test"
    (raw / "supplier_products").mkdir(parents=True, exist_ok=True)
    (raw / "suppliers").mkdir(parents=True, exist_ok=True)
    (raw / "inventory").mkdir(parents=True, exist_ok=True)

    sp_rows = [{
        "supplier_code": sc, "product_sku": "SKU-000",
        "unit_price": 10.0 + i, "moq": 10, "max_order_qty": 1000,
        "lead_time_days": 7.0, "lead_time_std_days": 1.5,
        "shipping_cost": 50.0, "is_preferred": i == 0,
    } for i, sc in enumerate(("SUP-A", "SUP-B"))]
    pd.DataFrame(sp_rows).to_parquet(raw / "supplier_products" / "data.parquet", index=False)

    pd.DataFrame([
        {"code": "SUP-A", "name": "A", "reliability_class": "standard",
         "cost_class": "standard", "quality_class": "standard",
         "lead_time_class": "standard", "capacity_class": "medium"},
        {"code": "SUP-B", "name": "B", "reliability_class": "standard",
         "cost_class": "standard", "quality_class": "standard",
         "lead_time_class": "standard", "capacity_class": "medium"},
    ]).to_parquet(raw / "suppliers" / "data.parquet", index=False)

    pd.DataFrame([{
        "product_sku": "SKU-000", "warehouse_code": "WH-A",
        "on_hand": 100.0, "reserved": 10.0, "inbound": 0.0,
    }]).to_parquet(raw / "inventory" / "data.parquet", index=False)

    return fdir, raw


def _stub_forecast(state, *, demand_checkpoint_dir, device="cpu"):
    from decision_engine.forecast_decision import ForecastResult

    H = state.horizon_days
    point = np.full(H, 20.0 * state.demand_multiplier, dtype=float)
    std = np.full(H, 3.0, dtype=float)
    origin = state.feature_frame["date"].iloc[-1]
    dates = [(pd.Timestamp(origin) + pd.Timedelta(days=i + 1)).date().isoformat()
             for i in range(H)]
    return ForecastResult(
        point=point, std=std, dates=dates,
        origin_date=str(origin)[:10],
        model_version="stub_demand_v0",
        feature_version=state.feature_version,
    )


def _stub_supplier_risk(state, *, supplier_checkpoint_dir):
    out = []
    for i, s in enumerate(state.suppliers):
        s.risk_probability = 0.1 + 0.05 * i
        s.risk_label = "low"
        out.append({
            "supplier_code": s.supplier_code,
            "risk_probability": s.risk_probability,
            "risk_label": s.risk_label,
            "reliability_class": s.reliability_class,
            "lead_time_days": float(s.lead_time_days),
            "lead_time_std_days": float(s.lead_time_std_days),
            "unit_price": float(s.unit_price),
            "moq": int(s.moq),
            "max_order_qty": int(s.max_order_qty),
            "shipping_cost": float(s.shipping_cost),
            "is_preferred": bool(s.is_preferred),
            "model_version": "stub_supplier_v0",
        })
    return out


@pytest.fixture()
def stub_engine(tmp_path, monkeypatch):
    """Return a DecisionEngine wired to stub models and tiny artifacts."""
    from decision_engine import forecast_decision, supplier_decision

    monkeypatch.setattr(forecast_decision, "produce_forecast", _stub_forecast)
    monkeypatch.setattr(supplier_decision, "score_suppliers", _stub_supplier_risk)

    curated, raw = _build_artifacts(tmp_path)

    from decision_engine.engine import DecisionEngine
    return DecisionEngine(
        demand_checkpoint_dir=tmp_path / "stub",
        supplier_checkpoint_dir=tmp_path / "stub",
        inventory_checkpoint_dir=None,
        curated_features_dir=curated,
        raw_dir=raw,
        simulation_samples=2000,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def test_catalog_has_expected_scenarios():
    from scenario_engine import SCENARIOS

    for key in (
        "demand_spike_10", "demand_spike_20", "demand_spike_50", "demand_spike_100",
        "lead_time_plus_1", "lead_time_plus_3", "lead_time_plus_7", "lead_time_plus_14",
        "supplier_failure", "capacity_reduction_50",
        "inventory_50", "inventory_100", "inventory_500",
        "inventory_1000", "inventory_5000", "inventory_10000",
        "budget_constraint",
        "service_target_90", "service_target_99",
    ):
        assert key in SCENARIOS, f"missing scenario: {key}"


# ---------------------------------------------------------------------------
# Runner (with stubs)
# ---------------------------------------------------------------------------
def test_demand_spike_increases_order(stub_engine):
    from scenario_engine import SCENARIOS, run_scenario

    spec = SCENARIOS["demand_spike_100"]
    result = run_scenario(
        spec, product_sku="SKU-000", warehouse_code="WH-A",
        engine=stub_engine, seed=42,
    )
    # A doubling of demand must not reduce the recommended order quantity.
    assert result.scenario["final_quantity"] >= result.baseline["final_quantity"] - 1e-6
    # And typically increases the total cost.
    assert result.scenario["total_cost"] >= result.baseline["total_cost"] - 1e-6


def test_lead_time_increase_raises_safety_stock(stub_engine):
    from scenario_engine import SCENARIOS, run_scenario

    spec = SCENARIOS["lead_time_plus_7"]
    result = run_scenario(
        spec, product_sku="SKU-000", warehouse_code="WH-A",
        engine=stub_engine, seed=42,
    )
    # Longer lead time => higher safety stock (or at least not lower).
    assert result.scenario["safety_stock"] >= result.baseline["safety_stock"] - 1e-6


def test_inventory_override_forces_state(stub_engine):
    from scenario_engine import SCENARIOS, run_scenario

    spec = SCENARIOS["inventory_5000"]
    result = run_scenario(
        spec, product_sku="SKU-000", warehouse_code="WH-A",
        engine=stub_engine, seed=42,
    )
    # 5000 units of inventory should drive stockout probability to ~0.
    assert result.scenario["stockout_probability"] <= 0.01


def test_scenario_is_reproducible(stub_engine):
    from scenario_engine import SCENARIOS, run_scenario

    spec = SCENARIOS["demand_spike_50"]
    r1 = run_scenario(spec, product_sku="SKU-000", warehouse_code="WH-A",
                     engine=stub_engine, seed=7)
    r2 = run_scenario(spec, product_sku="SKU-000", warehouse_code="WH-A",
                     engine=stub_engine, seed=7)
    for k in r1.delta:
        assert r1.delta[k] == r2.delta[k]


def test_different_scenarios_produce_different_deltas(stub_engine):
    from scenario_engine import SCENARIOS, run_scenario

    a = run_scenario(SCENARIOS["demand_spike_100"], product_sku="SKU-000",
                     warehouse_code="WH-A", engine=stub_engine, seed=1)
    b = run_scenario(SCENARIOS["inventory_50"], product_sku="SKU-000",
                     warehouse_code="WH-A", engine=stub_engine, seed=1)
    # A tiny inventory and a doubled demand must not produce identical deltas.
    assert a.delta["final_quantity"] != b.delta["final_quantity"]