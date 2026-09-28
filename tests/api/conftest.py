"""Fixtures for API tests.

Uses FastAPI TestClient bound to the real app. The DB dependency is
overridden with the transaction-scoped `db_session` fixture from the root
conftest, so no state leaks between tests.

The whole module is skipped if Postgres is not reachable.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(db_session, monkeypatch):
    """TestClient with dependency-overridden DB session."""
    from apps.api import dependencies as api_deps
    from apps.api.main import create_app

    app = create_app()

    # Override the get_db dependency with the transactional test session.
    def _override_get_db():
        yield db_session

    app.dependency_overrides[api_deps.get_db] = _override_get_db
    with TestClient(app) as c:
        yield c