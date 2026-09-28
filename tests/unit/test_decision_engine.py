"""Decision engine tests.

These use tiny synthetic artifacts so they can run without any trained
models (they mock the demand and supplier predictors). The end-to-end test
`test_decision_trace_integration` runs the real orchestrator with stub
artifacts produced on the fly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
def test_guardrails_detect_budget_violation():
    from decision_engine.policy_guardrails import check_guardrails

    r = check_guardrails(
        recommended_quantity=100.0,
        allocation_total_cost=5000.0,
        budget=1000.0,
        service_level_target=0.95,
        predicted_service_level=0.9,
        lead_time_days=7.0,
    )
    assert not r.passed
    assert any("budget" in v for v in r.violations)


def test_guardrails_detect_negative_quantity():
    from decision_engine.policy_guardrails import check_guardrails

    r = check_guardrails(
        recommended_quantity=-5.0,
        allocation_total_cost=0.0,
        budget=None,
        service_level_target=0.95,
        predicted_service_level=1.0,
        lead_time_days=7.0,
    )
    assert not r.passed


def test_guardrails_pass_on_valid_input():
    from decision_engine.policy_guardrails import check_guardrails

    r = check_guardrails(
        recommended_quantity=100.0,
        allocation_total_cost=500.0,
        budget=1000.0,
        service_level_target=0.95,
        predicted_service_level=0.96,
        lead_time_days=7.0,
    )
    assert r.passed


# ---------------------------------------------------------------------------
# Risk engine
# ---------------------------------------------------------------------------
def test_risk_multiplier_increases_with_risk():
    from decision_engine.risk_engine import risk_multiplier
    from decision_engine.state_builder import SupplierCandidate

    low = SupplierCandidate(None, "A", 10.0, 1, 1000, 50.0, 7.0, 1.0, False)
    low.risk_probability = 0.05
    high = SupplierCandidate(None, "B", 10.0, 1, 1000, 50.0, 7.0, 1.0, False)
    high.risk_probability = 0.40

    m_low = risk_multiplier(low, base_lead_time=7.0, horizon_days=30)
    m_high = risk_multiplier(high, base_lead_time=7.0, horizon_days=30)
    assert m_high > m_low


# ---------------------------------------------------------------------------
# Explanation engine
# ---------------------------------------------------------------------------
def test_explanation_uses_numeric_evidence():
    from decision_engine.explanation_engine import build_explanation

    trace = {
        "request": {"service_level": 0.95, "product_sku": "X", "warehouse_code": "Y"},
        "forecast_point": [10.0, 12.0, 11.0],
        "forecast_std": [1.0, 1.2, 0.9],
        "inventory_state": {"available": 50.0, "inventory_position": 70.0},
        "ltd_simulation": {"mean": 200.0, "std": 30.0},
        "stockout_probability": 0.35,
        "safety_stock": 40.0,
        "reorder_point": 240.0,
        "final_quantity": 170.0,
        "total_cost": 1950.0,
        "final_suppliers": [
            {"supplier_code": "SUP-1", "quantity": 170.0, "risk_probability": 0.1, "lead_time_days": 7.0},
        ],
        "expected_service_level": 0.65,
        "warnings": [],
        "dqn_action": {"multiplier": 1.25, "order_quantity": 150.0},
    }
    result = build_explanation(trace)
    text = result["explanation"]
    assert "200" in text or "200.00" in text
    assert "SUP-1" in text
    assert "65" in text or "0.65" in text
    assert len(result["evidence"]) > 5


# ---------------------------------------------------------------------------
# Confidence engine
# ---------------------------------------------------------------------------
def test_confidence_bounded_and_responds_to_risk():
    from decision_engine.confidence_engine import compute_confidence

    base = {
        "forecast_point": [10.0] * 10,
        "forecast_std": [1.0] * 10,
        "final_suppliers": [{"risk_probability": 0.05}],
        "allocation": {"requested_quantity": 100.0, "allocated_quantity": 100.0},
        "stockout_probability": 0.05,
    }
    high = compute_confidence(base)

    risky = dict(base)
    risky["final_suppliers"] = [{"risk_probability": 0.9}]
    risky["stockout_probability"] = 0.9
    low = compute_confidence(risky)

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low