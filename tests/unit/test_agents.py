"""Agent orchestration tests.

These use tiny synthetic artifacts so they exercise the full pipeline without
requiring trained models. Where a model is needed we stub the underlying
engine function using monkeypatch.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures — tiny artifacts
# ---------------------------------------------------------------------------
def _build_features(root: Path) -> None:
    from data_platform.features.feature_registry import FEATURE_VERSION

    rng = np.random.default_rng(0)
    start = date(2022, 1, 1)
    n_days = 120
    rows = []
    for sku in ("SKU-000", "SKU-001"):
        for d in range(n_days):
            dt = start + timedelta(days=d)
            q = float(max(0.0, 20.0 + rng.normal(0, 3)))
            row = {
                "product_sku": sku,
                "warehouse_code": "WH-A",
                "date": pd.Timestamp(dt),
                "quantity": q,
                "feat_demand_lag_1": q,
                "feat_demand_roll_std_28": 3.0,
                "feat_demand_roll_mean_7": q,
                "feat_demand_expand_mean": q,
                "feat_promo_count_7": 0,
                "feat_holiday_count_7": 0,
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
                "category_code": "CAT-A",
                "unit_cost": 10.0, "unit_price": 25.0,
                "base_demand": 20.0, "lead_time_days": 7,
            }
            rows.append(row)

    fdir = root / "curated" / "test" / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(fdir / f"demand_features_{FEATURE_VERSION}.parquet", index=False)

    # Minimal supplier feature table
    sup_rows = []
    for sku in ("SKU-000", "SKU-001"):
        for sc in ("SUP-A", "SUP-B"):
            for d in range(n_days):
                dt = start + timedelta(days=d)
                sup_rows.append({
                    "supplier_code": sc,
                    "product_sku": sku,
                    "date": pd.Timestamp(dt),
                    "feat_sup_delay_count_30": 0.0,
                    "feat_sup_delay_count_90": 0.0,
                    "feat_sup_quality_count_30": 0.0,
                    "feat_sup_outage_count_90": 0.0,
                    "feat_sup_ontime_proxy_30": 0.9,
                    "feat_sup_days_since_delay": 10.0,
                    "feat_sup_lead_time_days": 7.0,
                    "feat_sup_lead_time_std_days": 1.5,
                    "feat_sup_unit_price": 10.0,
                    "feat_sup_moq": 10.0,
                    "feat_sup_shipping_cost": 50.0,
                    "feat_sup_reliability": "standard",
                    "feat_sup_cost_class": "standard",
                    "feat_sup_quality_class": "standard",
                    "feat_sup_lead_time_class": "standard",
                    "feat_sup_capacity_class": "medium",
                })
    pd.DataFrame(sup_rows).to_parquet(
        fdir / f"supplier_features_{FEATURE_VERSION}.parquet", index=False
    )

    # Raw tables
    raw = root / "raw" / "test"
    (raw / "supplier_products").mkdir(parents=True, exist_ok=True)
    (raw / "suppliers").mkdir(parents=True, exist_ok=True)
    (raw / "inventory").mkdir(parents=True, exist_ok=True)

    sp_rows = []
    for sku in ("SKU-000", "SKU-001"):
        for i, sc in enumerate(("SUP-A", "SUP-B")):
            sp_rows.append({
                "supplier_code": sc, "product_sku": sku,
                "unit_price": 10.0 + i, "moq": 10, "max_order_qty": 1000,
                "lead_time_days": 7.0, "lead_time_std_days": 1.5,
                "shipping_cost": 50.0, "is_preferred": i == 0,
            })
    pd.DataFrame(sp_rows).to_parquet(raw / "supplier_products" / "data.parquet", index=False)

    sup_master = pd.DataFrame([
        {"code": "SUP-A", "name": "A", "reliability_class": "standard",
         "cost_class": "standard", "quality_class": "standard",
         "lead_time_class": "standard", "capacity_class": "medium"},
        {"code": "SUP-B", "name": "B", "reliability_class": "standard",
         "cost_class": "standard", "quality_class": "standard",
         "lead_time_class": "standard", "capacity_class": "medium"},
    ])
    sup_master.to_parquet(raw / "suppliers" / "data.parquet", index=False)

    inv_rows = [{
        "product_sku": sku, "warehouse_code": "WH-A",
        "on_hand": 100.0, "reserved": 10.0, "inbound": 0.0,
    } for sku in ("SKU-000", "SKU-001")]
    pd.DataFrame(inv_rows).to_parquet(raw / "inventory" / "data.parquet", index=False)


def _stub_forecast(state, *, demand_checkpoint_dir, device="cpu"):
    """Return a ForecastResult without needing a trained demand model."""
    from decision_engine.forecast_decision import ForecastResult

    horizon = state.horizon_days
    point = np.full(horizon, 20.0 * state.demand_multiplier, dtype=float)
    std = np.full(horizon, 3.0, dtype=float)
    dates = [
        (pd.Timestamp(state.feature_frame["date"].iloc[-1]) + pd.Timedelta(days=i + 1)).date().isoformat()
        for i in range(horizon)
    ]
    return ForecastResult(
        point=point, std=std, dates=dates,
        origin_date=str(state.feature_frame["date"].iloc[-1])[:10],
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


# ---------------------------------------------------------------------------
# Message schema
# ---------------------------------------------------------------------------
def test_agent_message_roundtrip():
    from agents.messages import AgentMessage

    m = AgentMessage(agent="demand", payload={"point": [1.0, 2.0]}, trace_id="t1")
    d = m.to_dict()
    assert d["agent"] == "demand"
    assert d["trace_id"] == "t1"
    assert d["payload"]["point"] == [1.0, 2.0]
    assert "message_id" in d and d["message_id"]


def test_agent_memory_records_and_retrieves():
    from agents.memory import AgentMemory
    from agents.messages import AgentMessage

    mem = AgentMemory()
    mem.start_trace("tX")
    mem.record(AgentMessage(agent="a", payload={}, trace_id="tX"))
    mem.record(AgentMessage(agent="b", payload={}, trace_id="tX"))
    rec = mem.get("tX")
    assert rec is not None
    assert len(rec.messages) == 2
    assert rec.by_agent("a")[0].agent == "a"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def test_orchestrator_with_stubbed_models(tmp_path, monkeypatch):
    from decision_engine import forecast_decision, supplier_decision

    monkeypatch.setattr(forecast_decision, "produce_forecast", _stub_forecast)
    monkeypatch.setattr(supplier_decision, "score_suppliers", _stub_supplier_risk)

    _build_features(tmp_path)

    from agents.coordinator_agent import AgentOrchestrator
    from decision_engine.decision_trace import DecisionRequest

    orch = AgentOrchestrator(
        demand_checkpoint_dir=tmp_path / "stub",
        supplier_checkpoint_dir=tmp_path / "stub",
        inventory_checkpoint_dir=None,
        curated_features_dir=tmp_path / "curated" / "test" / "features",
        raw_dir=tmp_path / "raw" / "test",
        simulation_samples=2_000,
    )
    req = DecisionRequest(
        product_sku="SKU-000", warehouse_code="WH-A", seed=42,
    )
    result = orch.run(req)

    # All five steps must have produced a message.
    agents = {m.agent for m in result.messages}
    assert {"demand", "supplier", "inventory", "procurement"} <= agents

    # Trace must be populated.
    trace = result.decision_trace
    assert trace.forecast_point and len(trace.forecast_point) == 30
    assert trace.supplier_risks
    assert trace.reorder_point > 0
    assert trace.safety_stock >= 0
    assert trace.explanation
    assert 0.0 <= trace.confidence <= 1.0


def test_orchestrator_is_reproducible(tmp_path, monkeypatch):
    from decision_engine import forecast_decision, supplier_decision

    monkeypatch.setattr(forecast_decision, "produce_forecast", _stub_forecast)
    monkeypatch.setattr(supplier_decision, "score_suppliers", _stub_supplier_risk)

    _build_features(tmp_path)

    from agents.coordinator_agent import AgentOrchestrator
    from decision_engine.decision_trace import DecisionRequest

    def run_once() -> tuple[float, float, float]:
        orch = AgentOrchestrator(
            demand_checkpoint_dir=tmp_path / "stub",
            supplier_checkpoint_dir=tmp_path / "stub",
            inventory_checkpoint_dir=None,
            curated_features_dir=tmp_path / "curated" / "test" / "features",
            raw_dir=tmp_path / "raw" / "test",
            simulation_samples=2_000,
        )
        trace = orch.run(DecisionRequest(
            product_sku="SKU-000", warehouse_code="WH-A", seed=42,
        )).decision_trace
        return trace.final_quantity, trace.total_cost, trace.stockout_probability

    a = run_once()
    b = run_once()
    assert a == b


def test_orchestrator_surfaces_errors_when_demand_agent_fails(tmp_path, monkeypatch):
    """If the demand agent fails, downstream agents must not fabricate output."""
    from decision_engine import forecast_decision, supplier_decision

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated demand failure")

    monkeypatch.setattr(forecast_decision, "produce_forecast", _boom)
    monkeypatch.setattr(supplier_decision, "score_suppliers", _stub_supplier_risk)

    _build_features(tmp_path)

    from agents.coordinator_agent import AgentOrchestrator
    from decision_engine.decision_trace import DecisionRequest

    orch = AgentOrchestrator(
        demand_checkpoint_dir=tmp_path / "stub",
        supplier_checkpoint_dir=tmp_path / "stub",
        inventory_checkpoint_dir=None,
        curated_features_dir=tmp_path / "curated" / "test" / "features",
        raw_dir=tmp_path / "raw" / "test",
        simulation_samples=2_000,
    )
    result = orch.run(DecisionRequest(product_sku="SKU-000", warehouse_code="WH-A", seed=42))

    msgs_by_agent = {m.agent: m for m in result.messages}
    assert msgs_by_agent["demand"].status == "error"
    # Inventory must also have failed (cannot fabricate a forecast).
    assert msgs_by_agent["inventory"].status == "error"
    # Procurement must have failed too.
    assert msgs_by_agent["procurement"].status == "error"
    # Supplier agent still succeeded.
    assert msgs_by_agent["supplier"].status != "error"


# ---------------------------------------------------------------------------
# Audit persistence (in-memory only — the DB test requires Postgres, which
# the base test suite skips when not available).
# ---------------------------------------------------------------------------
def test_agent_message_summary_shape():
    from agents.messages import AgentMessage

    m = AgentMessage(agent="x", payload={"a": 1, "b": 2}, trace_id="t")
    s = m.summary()
    assert s["agent"] == "x"
    assert s["n_payload_keys"] == 2
    assert s["status"] == "ok"