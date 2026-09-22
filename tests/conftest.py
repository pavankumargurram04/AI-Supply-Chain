"""Shared pytest fixtures.

Two layers:
- unit-level fixtures (no DB) — used by tests/unit/*
- integration fixtures (Postgres) — used by tests/integration/*

Integration tests are skipped automatically if Postgres is unreachable,
so `pytest -q` works on a laptop without Docker running.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

# Ensure project root on sys.path.
ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine.url import make_url  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from core.config import get_settings  # noqa: E402


# --- Unit fixtures ---------------------------------------------------------
@pytest.fixture(scope="session")
def settings():
    return get_settings()


# --- Integration fixtures --------------------------------------------------
def _pg_reachable() -> bool:
    try:
        url = make_url(get_settings().database_url)
        admin = url.set(database="postgres")
        engine = create_engine(admin, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def pg_available() -> bool:
    return _pg_reachable()


@pytest.fixture(scope="session")
def test_db_url(pg_available: bool) -> str:
    if not pg_available:
        pytest.skip("PostgreSQL not reachable; skipping integration tests.")
    url = make_url(get_settings().database_url)
    return url.set(database="supplychain_test").render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def test_engine(test_db_url: str):
    """Create (or recreate) a dedicated test database and run migrations."""
    from alembic import command
    from alembic.config import Config

    admin = make_url(test_db_url).set(database="postgres")
    admin_engine = create_engine(admin, isolation_level="AUTOCOMMIT", future=True)
    with admin_engine.connect() as conn:
        target = make_url(test_db_url).database
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": target}
        ).scalar_one_or_none()
        if not exists:
            safe = target.replace('"', '""')
            conn.execute(text(f'CREATE DATABASE "{safe}"'))
    admin_engine.dispose()

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "db" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", test_db_url)

    # Rebuild from scratch for test determinism.
    from sqlalchemy import create_engine as _ce
    eng = _ce(test_db_url, future=True)
    with eng.begin() as conn:
        conn.execute(text('DROP SCHEMA public CASCADE'))
        conn.execute(text('CREATE SCHEMA public'))
    eng.dispose()

    command.upgrade(cfg, "head")

    engine = _ce(test_db_url, future=True)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(test_engine) -> Iterator[Session]:
    """Session bound to a transaction that is rolled back after the test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection, expire_on_commit=False, future=True)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
