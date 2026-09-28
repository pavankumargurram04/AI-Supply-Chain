"""Authentication: login, refresh, current user."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db
from apps.api.schemas import LoginRequest, TokenResponse, UserOut
from core.config import get_settings
from core.errors import AuthenticationError
from core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from db.repositories.user_repository import UserRepository

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    repo = UserRepository(db)
    user = repo.get_by_username(payload.username) or repo.get_by_email(payload.username.lower())
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise AuthenticationError("Invalid username or password")
    if not user.is_active:
        raise AuthenticationError("User is inactive")

    user.last_login_at = datetime.now(tz=timezone.utc)
    db.flush()

    settings = get_settings()
    access = create_access_token(user.id, role=user.role)
    refresh = create_refresh_token(user.id)
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(refresh_token: str, db: Session = Depends(get_db)) -> TokenResponse:
    payload = decode_token(refresh_token)
    if payload.get("typ") != "refresh":
        raise AuthenticationError("Not a refresh token")
    user_id = int(payload["sub"])
    repo = UserRepository(db)
    user = repo.get(user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("User not found or inactive")
    settings = get_settings()
    return TokenResponse(
        access_token=create_access_token(user.id, role=user.role),
        refresh_token=create_refresh_token(user.id),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserOut)
def me(current = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current)