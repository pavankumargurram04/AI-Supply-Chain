"""Repository for User."""

from __future__ import annotations

from typing import Optional

from db.models.user import User
from db.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    def get_by_email(self, email: str) -> Optional[User]:
        return self.get_by(email=email.lower())

    def get_by_username(self, username: str) -> Optional[User]:
        return self.get_by(username=username)

    def create(
        self,
        *,
        email: str,
        username: str,
        hashed_password: str,
        role: str,
        full_name: Optional[str] = None,
    ) -> User:
        user = User(
            email=email.lower(),
            username=username,
            hashed_password=hashed_password,
            role=role,
            full_name=full_name,
        )
        return self.add(user)