"""User administration (admin only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.dependencies import get_db, require_admin
from apps.api.schemas import UserOut
from core.enums import UserRole
from core.errors import ConflictError, ValidationError
from core.security import hash_password
from db.repositories.user_repository import UserRepository

router = APIRouter(prefix="/users", tags=["users"])


from pydantic import BaseModel, EmailStr


class CreateUserIn(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: str = UserRole.ANALYST.value
    full_name: str | None = None


@router.get("", response_model=list[UserOut], dependencies=[Depends(require_admin)])
def list_users(db: Session = Depends(get_db)) -> list[UserOut]:
    repo = UserRepository(db)
    return [UserOut.model_validate(u) for u in repo.list(limit=500)]


@router.post("", response_model=UserOut, status_code=201,
             dependencies=[Depends(require_admin)])
def create_user(payload: CreateUserIn, db: Session = Depends(get_db)) -> UserOut:
    if payload.role not in {r.value for r in UserRole}:
        raise ValidationError(f"Unknown role: {payload.role}")
    repo = UserRepository(db)
    if repo.get_by_username(payload.username) is not None:
        raise ConflictError("username already exists")
    if repo.get_by_email(payload.email.lower()) is not None:
        raise ConflictError("email already exists")
    user = repo.create(
        email=payload.email,
        username=payload.username,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        full_name=payload.full_name,
    )
    return UserOut.model_validate(user)