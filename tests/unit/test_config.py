"""Sanity tests for configuration and determinism."""

from __future__ import annotations

from core.config import get_settings
from core.deterministic import derive_seed, seed_everything


def test_settings_singleton_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b


def test_database_url_is_well_formed():
    url = get_settings().database_url
    assert url.startswith("postgresql+psycopg2://")
    assert "@" in url


def test_redis_url_is_well_formed():
    url = get_settings().redis_url
    assert url.startswith("redis://")


def test_seed_everything_returns_seed():
    assert seed_everything(123) == 123


def test_derive_seed_is_deterministic_and_distinct():
    a = derive_seed(42, "demand", "SKU-1")
    b = derive_seed(42, "demand", "SKU-1")
    c = derive_seed(42, "supplier", "SUP-9")
    assert a == b
    assert a != c