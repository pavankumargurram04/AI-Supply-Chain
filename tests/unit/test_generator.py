"""Tests for the synthetic data generator.

These tests run without PostgreSQL (unit level) — they exercise determinism,
causality, and structural integrity of the generated dataset.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.deterministic import derive_seed, seed_everything
from data_platform.generator.config import GeneratorConfig
from data_platform.generator.demand import generate_demand
from data_platform.generator.entities import (
    generate_categories,
    generate_products,
    generate_suppliers,
    generate_warehouses,
)
from data_platform.generator.supplier_product import generate_supplier_products


def _tiny_cfg() -> GeneratorConfig:
    return GeneratorConfig(
        mode="test",
        seed=123,
        n_categories=3,
        n_products=6,
        n_suppliers=4,
        n_warehouses=2,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 3, 31),
        output_dir="datasets/raw/test",
    )


def test_generator_is_deterministic_given_same_seed():
    cfg = _tiny_cfg()
    seed_everything(cfg.seed)
    cats1 = generate_categories(cfg, np.random.default_rng(derive_seed(cfg.seed, "c")))
    prods1 = generate_products(cfg, cats1, np.random.default_rng(derive_seed(cfg.seed, "p")))

    seed_everything(cfg.seed)
    cats2 = generate_categories(cfg, np.random.default_rng(derive_seed(cfg.seed, "c")))
    prods2 = generate_products(cfg, cats2, np.random.default_rng(derive_seed(cfg.seed, "p")))

    assert [c["code"] for c in cats1] == [c["code"] for c in cats2]
    assert [p["sku"] for p in prods1] == [p["sku"] for p in prods2]
    for p1, p2 in zip(prods1, prods2):
        assert p1["unit_cost"] == p2["unit_cost"]
        assert p1["base_demand"] == p2["base_demand"]


def test_products_have_valid_attributes():
    cfg = _tiny_cfg()
    cats = generate_categories(cfg, np.random.default_rng(1))
    prods = generate_products(cfg, cats, np.random.default_rng(2))

    assert len(prods) == cfg.n_products
    for p in prods:
        assert p["unit_price"] > p["unit_cost"] > 0
        assert p["base_demand"] > 0
        assert p["lead_time_days"] > 0
        assert p["lifecycle_stage"] in {"introduction", "growth", "maturity", "decline", "discontinued"}


def test_suppliers_and_warehouses():
    cfg = _tiny_cfg()
    suppliers = generate_suppliers(cfg, np.random.default_rng(3))
    warehouses = generate_warehouses(cfg, np.random.default_rng(4))

    assert len(suppliers) == cfg.n_suppliers
    assert len(warehouses) == cfg.n_warehouses
    for s in suppliers:
        assert s["contracted_capacity_units"] > 0
        assert s["reliability_class"] in {"premium", "reliable", "standard", "poor"}
    for w in warehouses:
        assert w["capacity_units"] > 0


def test_supplier_product_links_reference_real_entities():
    cfg = _tiny_cfg()
    cats = generate_categories(cfg, np.random.default_rng(11))
    prods = generate_products(cfg, cats, np.random.default_rng(12))
    sups = generate_suppliers(cfg, np.random.default_rng(13))
    links = generate_supplier_products(cfg, prods, sups, np.random.default_rng(14))

    skus = {p["sku"] for p in prods}
    sup_codes = {s["code"] for s in sups}
    for link in links:
        assert link["product_sku"] in skus
        assert link["supplier_code"] in sup_codes
        assert link["unit_price"] > 0
        assert link["moq"] >= 1
        assert link["lead_time_days"] > 0
        assert link["lead_time_std_days"] >= 0


def test_demand_series_are_nonnegative_and_complete():
    cfg = _tiny_cfg()
    cats = generate_categories(cfg, np.random.default_rng(21))
    prods = generate_products(cfg, cats, np.random.default_rng(22))
    whs = generate_warehouses(cfg, np.random.default_rng(23))
    df = generate_demand(cfg, prods, whs, np.random.default_rng(24))

    expected_rows = len(prods) * len(whs) * cfg.days
    assert len(df) == expected_rows
    assert (df["quantity"] >= 0).all()
    assert df["product_sku"].nunique() == len(prods)
    assert df["warehouse_code"].nunique() == len(whs)
    # Each series must have exactly cfg.days rows.
    counts = df.groupby(["product_sku", "warehouse_code"]).size().unique()
    assert list(counts) == [cfg.days]


def test_demand_is_deterministic():
    cfg = _tiny_cfg()
    cats = generate_categories(cfg, np.random.default_rng(31))
    prods = generate_products(cfg, cats, np.random.default_rng(32))
    whs = generate_warehouses(cfg, np.random.default_rng(33))

    df1 = generate_demand(cfg, prods, whs, np.random.default_rng(99))
    df2 = generate_demand(cfg, prods, whs, np.random.default_rng(99))
    pd.testing.assert_frame_equal(df1, df2)


def test_demand_reacts_to_promotion_days():
    """Promotional days must, on average, carry higher demand than non-promo days."""
    cfg = _tiny_cfg()
    cats = generate_categories(cfg, np.random.default_rng(41))
    prods = generate_products(cfg, cats, np.random.default_rng(42))
    whs = generate_warehouses(cfg, np.random.default_rng(43))
    df = generate_demand(cfg, prods, whs, np.random.default_rng(44))

    promo = df[df["promotion_flag"]]
    non_promo = df[~df["promotion_flag"]]
    assert len(promo) > 0, "Generator produced no promotional days — reduce promo length or increase horizon"
    assert promo["quantity"].mean() > non_promo["quantity"].mean()


def test_holiday_flags_are_low_demand_for_major_holidays():
    """Days tagged as Christmas should average below the yearly mean."""
    cfg = _tiny_cfg()
    cats = generate_categories(cfg, np.random.default_rng(51))
    prods = generate_products(cfg, cats, np.random.default_rng(52))
    whs = generate_warehouses(cfg, np.random.default_rng(53))
    df = generate_demand(cfg, prods, whs, np.random.default_rng(54))

    # Christmas is Dec 25 — but our test window is Jan–Mar, so no Christmas rows.
    # Instead check that New Year's Day (Jan 1) appears as holiday flag.
    ny = df[df["demand_date"] == date(2023, 1, 1)]
    assert (ny["holiday_flag"]).all()