"""Smoke tests for the core API endpoints.

These tests exercise the auth flow, master-data listing, and error handling.
They require Postgres (skipped otherwise).
"""

from __future__ import annotations

import pytest

from core.enums import UserRole
from core.security import hash_password

pytestmark = pytest.mark.integration


def _seed_user(db_session, *, username="alice", password="pw123", role=UserRole.ANALYST.value):
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
# Health (no auth)
# ---------------------------------------------------------------------------
def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_returns_checks(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert "checks" in body
    assert "database" in body["checks"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def test_login_with_wrong_password_returns_401(client, db_session):
    _seed_user(db_session)
    r = client.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong"})
    assert r.status_code == 401


def test_login_success_and_me(client, db_session):
    _seed_user(db_session)
    token = _login(client, "alice", "pw123")
    r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["username"] == "alice"


def test_protected_route_requires_token(client):
    r = client.get("/api/v1/products")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Products / suppliers / warehouses (require auth)
# ---------------------------------------------------------------------------
def test_products_listing_with_token(client, db_session):
    _seed_user(db_session, username="bob", password="pw", role=UserRole.ANALYST.value)
    token = _login(client, "bob", "pw")
    r = client.get("/api/v1/products", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_get_nonexistent_product_returns_404(client, db_session):
    _seed_user(db_session, username="carol", password="pw", role=UserRole.ANALYST.value)
    token = _login(client, "carol", "pw")
    r = client.get("/api/v1/products/999999999", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 404


def test_suppliers_listing(client, db_session):
    _seed_user(db_session, username="dave", password="pw", role=UserRole.ANALYST.value)
    token = _login(client, "dave", "pw")
    r = client.get("/api/v1/suppliers", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_warehouses_listing(client, db_session):
    _seed_user(db_session, username="eve", password="pw", role=UserRole.ANALYST.value)
    token = _login(client, "eve", "pw")
    r = client.get("/api/v1/warehouses", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Request-ID propagation
# ---------------------------------------------------------------------------
def test_request_id_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == "abc123"


# ---------------------------------------------------------------------------
# Admin-only route
# ---------------------------------------------------------------------------
def test_non_admin_cannot_list_users(client, db_session):
    _seed_user(db_session, username="frank", password="pw", role=UserRole.ANALYST.value)
    token = _login(client, "frank", "pw")
    r = client.get("/api/v1/users", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403


def test_admin_can_list_users(client, db_session):
    _seed_user(db_session, username="root", password="pw", role=UserRole.ADMIN.value)
    token = _login(client, "root", "pw")
    r = client.get("/api/v1/users", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200