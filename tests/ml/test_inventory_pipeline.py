"""Tests for the inventory DQN, baselines, and allocation."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def test_simulation_deterministic():
    from ml.inventory.simulation import SimulationConfig, simulate_lead_time_demand

    fc = np.linspace(20, 25, 30)
    res = np.random.default_rng(0).normal(0, 3, 500)
    cfg = SimulationConfig(seed=42, n_samples=3000, alpha=0.95)
    a = simulate_lead_time_demand(forecast_point=fc, forecast_residuals=res,
                                  lead_time_mean=7.0, lead_time_std=1.5,
                                  available_inventory=120.0, config=cfg)
    b = simulate_lead_time_demand(forecast_point=fc, forecast_residuals=res,
                                  lead_time_mean=7.0, lead_time_std=1.5,
                                  available_inventory=120.0, config=cfg)
    assert a.mean == b.mean
    assert a.quantiles[0.95] == b.quantiles[0.95]


def test_simulation_stockout_probability_increases_as_inventory_decreases():
    from ml.inventory.simulation import SimulationConfig, simulate_lead_time_demand

    fc = np.full(30, 20.0)
    res = np.random.default_rng(0).normal(0, 4, 500)
    cfg = SimulationConfig(seed=1, n_samples=4000)
    high_inv = simulate_lead_time_demand(
        forecast_point=fc, forecast_residuals=res,
        lead_time_mean=7.0, lead_time_std=1.0,
        available_inventory=400.0, config=cfg,
    )
    low_inv = simulate_lead_time_demand(
        forecast_point=fc, forecast_residuals=res,
        lead_time_mean=7.0, lead_time_std=1.0,
        available_inventory=60.0, config=cfg,
    )
    assert low_inv.stockout_probability > high_inv.stockout_probability


def test_safety_stock_positive_at_high_service_level():
    from ml.inventory.safety_stock import safety_stock_from_simulation
    from ml.inventory.simulation import SimulationConfig, simulate_lead_time_demand

    fc = np.full(30, 20.0)
    res = np.random.default_rng(0).normal(0, 4, 500)
    cfg = SimulationConfig(seed=1, n_samples=4000, alpha=0.95)
    r = simulate_lead_time_demand(
        forecast_point=fc, forecast_residuals=res,
        lead_time_mean=7.0, lead_time_std=1.0,
        available_inventory=400.0, config=cfg,
    )
    ss = safety_stock_from_simulation(r, service_level=0.95)
    assert ss >= 0.0


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def test_cost_components_are_additive():
    from ml.inventory.cost_model import CostParameters, total_cost
    p = CostParameters(unit_cost=10.0, holding_rate_annual=0.20,
                       order_fixed_cost=100.0, shortage_cost_per_unit=30.0,
                       shipping_cost_per_order=50.0)
    c = total_cost(average_inventory_units=100.0, n_orders=2, unmet_units=5.0,
                   params=p, days=30)
    assert c["total"] == pytest.approx(c["holding"] + c["ordering"] + c["shortage"])
    assert c["shortage"] == pytest.approx(5.0 * 30.0)
    assert c["ordering"] == pytest.approx(2 * (100.0 + 50.0))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def test_environment_respects_nonnegative_inventory():
    from ml.inventory.cost_model import CostParameters
    from ml.inventory.environment import EnvConfig, InventoryEnvironment

    p = CostParameters(unit_cost=10.0, holding_rate_annual=0.2,
                       order_fixed_cost=100.0, shortage_cost_per_unit=30.0,
                       shipping_cost_per_order=50.0)
    env = InventoryEnvironment(EnvConfig(cost=p, episode_days=20))
    T = 60
    env.attach_series(
        forecast_point=np.full(T, 20.0, dtype=np.float32),
        forecast_std=np.full(T, 4.0, dtype=np.float32),
        actual_demand=np.full(T, 20.0, dtype=np.float32),
        lead_time_mean=np.full(T, 7.0, dtype=np.float32),
        lead_time_std=np.full(T, 1.5, dtype=np.float32),
        supplier_risk=np.full(T, 0.1, dtype=np.float32),
        initial_inventory=50.0, seed=0,
    )
    state = env.reset(start_index=0)
    done = False
    while not done:
        state, r, done, info = env.step(0)   # never order
        assert env.on_hand >= 0.0


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
def test_replay_buffer_capacity_and_sample():
    from ml.inventory.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=50, state_dim=11)
    rng = np.random.default_rng(0)
    for i in range(100):
        s = rng.normal(size=11).astype(np.float32)
        buf.push(s, action=i % 5, reward=float(i), next_state=s, done=False)
    assert len(buf) == 50   # capped
    s, a, r, ns, d = buf.sample(batch_size=16, rng=rng)
    assert s.shape == (16, 11)
    assert a.shape == (16,)


# ---------------------------------------------------------------------------
# DQN training smoke
# ---------------------------------------------------------------------------
def test_dqn_training_tiny_run(tmp_path):
    from ml.inventory.train import DQNTrainConfig, train_inventory_dqn

    # Build a tiny demand feature table (reuse the same recipe as in the
    # demand test, but only the columns _load_series needs).
    import pandas as pd
    from datetime import date, timedelta
    from data_platform.features.feature_registry import FEATURE_VERSION

    rng = np.random.default_rng(0)
    start = date(2022, 1, 1)
    rows = []
    for s_i in range(2):
        sku = f"SKU-{s_i:03d}"
        for d in range(120):
            dt = start + timedelta(days=d)
            q = float(max(0.0, 20.0 + rng.normal(0, 3)))
            rows.append({
                "product_sku": sku,
                "warehouse_code": "WH-A",
                "date": pd.Timestamp(dt),
                "quantity": q,
                "feat_demand_lag_1": q,
                "feat_demand_roll_std_28": 3.0,
                "unit_cost": 10.0,
            })
    df = pd.DataFrame(rows)
    fdir = tmp_path / "features"
    fdir.mkdir()
    df.to_parquet(fdir / f"demand_features_{FEATURE_VERSION}.parquet", index=False)

    cfg = DQNTrainConfig(
        features_dir=str(fdir),
        output_dir=str(tmp_path / "artifact"),
        seed=1,
        episode_days=20,
        n_episodes=3,
        max_series=2,
        batch_size=16,
        replay_capacity=2_000,
        warmup_steps=100,
        target_update_steps=50,
        hidden=32,
        n_layers=1,
        device="cpu",
    )
    meta = train_inventory_dqn(cfg)
    out = tmp_path / "artifact"
    assert (out / "dqn_best.pt").exists() or (out / "dqn_last.pt").exists()
    assert (out / "training_metadata.json").exists()
    assert np.isfinite(meta["final_eval_cost"])


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def test_baselines_run_and_produce_metrics():
    from ml.inventory.baselines import ALL_BASELINES, run_baseline
    from ml.inventory.cost_model import CostParameters

    p = CostParameters(unit_cost=10.0, holding_rate_annual=0.2,
                       order_fixed_cost=100.0, shortage_cost_per_unit=30.0,
                       shipping_cost_per_order=50.0)
    T = 90
    rng = np.random.default_rng(0)
    series = {
        "forecast_point": np.full(T, 20.0, dtype=np.float32),
        "forecast_std": np.full(T, 4.0, dtype=np.float32),
        "actual_demand": np.clip(rng.normal(20, 4, size=T).astype(np.float32), 0, None),
        "lead_time_mean": np.full(T, 7.0, dtype=np.float32),
        "lead_time_std": np.full(T, 1.5, dtype=np.float32),
        "supplier_risk": np.full(T, 0.1, dtype=np.float32),
    }
    for name in ALL_BASELINES:
        r = run_baseline(name, series, cost=p, episode_days=30, seed=0)
        assert np.isfinite(r.total_cost)
        assert 0.0 <= r.service_level <= 1.0


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
def test_allocation_respects_moq_and_capacity():
    from ml.inventory.allocation import SupplierOffer, allocate_suppliers

    offers = [
        SupplierOffer(1, "SUP-A", 10.0, moq=50, max_order_qty=200,
                      shipping_cost=100.0, lead_time_days=7, risk_probability=0.1),
        SupplierOffer(2, "SUP-B", 9.5, moq=100, max_order_qty=300,
                      shipping_cost=150.0, lead_time_days=10, risk_probability=0.3),
    ]
    result = allocate_suppliers(required_quantity=250.0, offers=offers)
    # Both constraints are respected.
    assert result.constraints["quantity_satisfied"] is True
    for l in result.lines:
        offer = next(o for o in offers if o.supplier_id == l.supplier_id)
        assert l.quantity >= offer.moq
        assert l.quantity <= offer.max_order_qty + 1e-6


def test_allocation_fails_gracefully_when_moq_too_high():
    from ml.inventory.allocation import SupplierOffer, allocate_suppliers

    offers = [
        SupplierOffer(1, "SUP-A", 10.0, moq=500, max_order_qty=1000,
                      shipping_cost=100.0, lead_time_days=7, risk_probability=0.1),
    ]
    result = allocate_suppliers(required_quantity=50.0, offers=offers)
    assert result.allocated_quantity == 0.0
    assert result.constraints["quantity_satisfied"] is False


def test_allocation_prefers_preferred_supplier_when_close_in_score():
    from ml.inventory.allocation import SupplierOffer, allocate_suppliers

    offers = [
        SupplierOffer(1, "SUP-A", 10.0, moq=10, max_order_qty=1000,
                      shipping_cost=100.0, lead_time_days=7, risk_probability=0.1),
        SupplierOffer(2, "SUP-B", 10.1, moq=10, max_order_qty=1000,
                      shipping_cost=100.0, lead_time_days=7, risk_probability=0.1),
    ]
    result = allocate_suppliers(
        required_quantity=100.0, offers=offers,
        preferred_supplier_ids=[2],
    )
    assert result.lines[0].supplier_id == 2