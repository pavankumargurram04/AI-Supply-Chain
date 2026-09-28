"""Reusable FastAPI dependencies: DB session, current user, role guards."""

from __future__ import annotations

from typing import Iterable, Optional

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from core.database import get_db as _get_db
from core.enums import UserRole
from core.errors import AuthenticationError, AuthorizationError
from core.security import decode_token
from db.models.user import User


def get_db() -> Session:
    """Alias for core.database.get_db (kept so routers import from one place)."""
    yield from _get_db()


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def get_current_user(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> User:
    token = _extract_bearer(authorization)
    if token is None:
        raise AuthenticationError("Missing bearer token")
    payload = decode_token(token)
    if payload.get("typ") != "access":
        raise AuthenticationError("Not an access token")
    try:
        user_id = int(payload.get("sub"))
    except (TypeError, ValueError) as e:
        raise AuthenticationError("Invalid subject in token") from e

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("User not found or inactive")
    return user


def require_roles(*roles: str):
    """Return a dependency that requires the current user to have one of `roles`."""
    allowed = set(roles)

    def _dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise AuthorizationError(
                f"Role {user.role!r} not permitted; requires one of {sorted(allowed)}"
            )
        return user

    return _dep


require_admin = require_roles(UserRole.ADMIN.value)
require_analyst = require_roles(
    UserRole.ADMIN.value,
    UserRole.ANALYST.value,
    UserRole.SUPPLY_CHAIN_MANAGER.value,
)
require_supply_chain_manager = require_roles(
    UserRole.ADMIN.value,
    UserRole.SUPPLY_CHAIN_MANAGER.value,
)
require_procurement_manager = require_roles(
    UserRole.ADMIN.value,
    UserRole.PROCUREMENT_MANAGER.value,
    UserRole.SUPPLY_CHAIN_MANAGER.value,
)
require_reviewer = require_roles(
    UserRole.ADMIN.value,
    UserRole.REVIEWER.value,
    UserRole.PROCUREMENT_MANAGER.value,
    UserRole.SUPPLY_CHAIN_MANAGER.value,
)