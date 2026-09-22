"""Password hashing and JWT utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import get_settings
from core.errors import AuthenticationError


_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------------- Passwords ----------------
def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd_ctx.verify(plain, hashed)
    except ValueError:
        return False


# ---------------- JWT ----------------
def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def create_access_token(
    subject: str | int,
    *,
    role: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
    expires_minutes: Optional[int] = None,
) -> str:
    settings = get_settings()
    expire_minutes = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(_now().timestamp()),
        "exp": int((_now() + timedelta(minutes=expire_minutes)).timestamp()),
        "typ": "access",
    }
    if role:
        payload["role"] = role
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(
    subject: str | int,
    *,
    expires_minutes: Optional[int] = None,
) -> str:
    settings = get_settings()
    expire_minutes = expires_minutes or settings.REFRESH_TOKEN_EXPIRE_MINUTES
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(_now().timestamp()),
        "exp": int((_now() + timedelta(minutes=expire_minutes)).timestamp()),
        "typ": "refresh",
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except JWTError as exc:
        raise AuthenticationError(f"Invalid token: {exc}") from exc
    return payload