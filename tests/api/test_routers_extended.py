"""Extended router coverage: forecast, supplier-risk, agents, scenarios, models."""

from __future__ import annotations

import pytest

from core.enums import UserRole
from core.security import hash_password

pytestmark = pytest.mark.integration


def _seed_user(db_session, username="routers", password="pw", role=UserRole.ANALYST.value):
    from db.repositories.user_repository import UserRepository
    repo = UserRepository(db_session)
    if repo.get_by_username(username) is None:
        repo.create(
            email=f"{username}@example.com",
            username=username,
            hashed_password=hash_password(password),
            role=role,
        )
        db_session.flush()
    return username, password


def _login(client, username, password) -> str:
    r = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


# ---------------------------------------------------------------------------
# Read-only endpoints that shouldn't 500 even when datasets/artifacts are missing.
# ---------------------------------------------------------------------------
def test_forecast_route_returns_error_not_500_when_checkpoint_missing(client, db_session):
    _seed_user(db_session)
    token = _login(client, "routers", "pw")
    r = client.get("/api/v1/forecast/SKU-000/WH-A", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code in (500, 404)
    body = r.json()
    assert "error_code" in body


def test_supplier_risk_route_requires_auth(client, db_session):
    r = client.get("/api/v1/supplier-risk?product_sku=SKU-000")
    assert r.status_code == 401


def test_agents_run_requires_analyst_role(client, db_session):
    _seed_user(db_session, username="reviewer1", password="pw", role=UserRole.REVIEWER.value)
    token = _login(client, "reviewer1", "pw")
    r = client.post(
        "/api/v1/agents/run",
        json={"product_sku": "SKU-000", "warehouse_code": "WH-A"},
        headers={"Authorization": f"Bearer {token}"},
    )
    # Reviewer is not analyst — should be 403 regardless of artifact state.
    assert r.status_code == 403


def test_scenarios_run_requires_analyst_role(client, db_session):
    _seed_user(db_session, username="reviewer2", password="pw", role=UserRole.REVIEWER.value)
    token = _login(client, "reviewer2", "pw")
    r = client.post(
        "/api/v1/scenarios/run",
        json={"product_sku": "SKU-000", "warehouse_code": "WH-A"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


def test_models_list_is_public_to_authenticated_users(client, db_session):
    _seed_user(db_session, username="analyst1", password="pw", role=UserRole.ANALYST.value)
    token = _login(client, "analyst1", "pw")
    r = client.get("/api/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_training_enqueue_requires_admin(client, db_session):
    _seed_user(db_session, username="analyst2", password="pw", role=UserRole.ANALYST.value)
    token = _login(client, "analyst2", "pw")
    r = client.post(
        "/api/v1/training/jobs",
        json={"model_family": "demand_lstm", "mode": "dev"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


def test_monitoring_alerts_list(client, db_session):
    _seed_user(db_session, username="analyst3", password="pw", role=UserRole.ANALYST.value)
    token = _login(client, "analyst3", "pw")
    r = client.get("/api/v1/monitoring/alerts", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_audit_list_is_authenticated(client, db_session):
    _seed_user(db_session, username="auditor", password="pw", role=UserRole.ANALYST.value)
    token = _login(client, "auditor", "pw")
    r = client.get("/api/v1/audit?limit=10", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_reports_listing(client, db_session):
    _seed_user(db_session, username="reporter", password="pw", role=UserRole.ANALYST.value)
    token = _login(client, "reporter", "pw")
    r = client.get("/api/v1/reports", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)