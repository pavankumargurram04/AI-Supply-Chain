"""Generic typed repository built on SQLAlchemy 2.0.

All concrete repositories inherit from this class and only add domain-specific
query methods. No hidden caching; no silent commits. The repository holds a
session that the caller owns.
"""

from __future__ import annotations

from typing import Any, Generic, Iterable, Optional, Sequence, Type, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from core.errors import ConflictError, NotFoundError, ValidationError
from db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    model: Type[ModelT]

    def __init__(self, session: Session) -> None:
        if not hasattr(self, "model"):
            raise TypeError(
                f"{type(self).__name__} must define a `model` class attribute."
            )
        self.session = session

    # ---------------- Read ----------------
    def get(self, pk: int) -> Optional[ModelT]:
        return self.session.get(self.model, pk)

    def get_or_raise(self, pk: int) -> ModelT:
        obj = self.get(pk)
        if obj is None:
            raise NotFoundError(
                f"{self.model.__name__} id={pk} not found",
                details={"model": self.model.__name__, "id": pk},
            )
        return obj

    def get_by(self, **filters: Any) -> Optional[ModelT]:
        stmt = select(self.model)
        for k, v in filters.items():
            if not hasattr(self.model, k):
                raise ValidationError(f"Unknown filter field: {k}")
            stmt = stmt.where(getattr(self.model, k) == v)
        return self.session.execute(stmt.limit(1)).scalar_one_or_none()

    def list(
        self,
        *,
        filters: Optional[dict[str, Any]] = None,
        order_by: Optional[Any] = None,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> Sequence[ModelT]:
        limit = max(1, min(int(limit), MAX_PAGE_SIZE))
        offset = max(0, int(offset))
        stmt = select(self.model)
        if filters:
            for k, v in filters.items():
                if not hasattr(self.model, k):
                    raise ValidationError(f"Unknown filter field: {k}")
                stmt = stmt.where(getattr(self.model, k) == v)
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        else:
            stmt = stmt.order_by(self.model.id)  # type: ignore[attr-defined]
        stmt = stmt.limit(limit).offset(offset)
        return list(self.session.execute(stmt).scalars().all())

    def count(self, *, filters: Optional[dict[str, Any]] = None) -> int:
        stmt = select(func.count()).select_from(self.model)
        if filters:
            for k, v in filters.items():
                if not hasattr(self.model, k):
                    raise ValidationError(f"Unknown filter field: {k}")
                stmt = stmt.where(getattr(self.model, k) == v)
        return int(self.session.execute(stmt).scalar_one())

    # ---------------- Write ----------------
    def add(self, obj: ModelT) -> ModelT:
        self.session.add(obj)
        self.session.flush()
        return obj

    def add_all(self, objs: Iterable[ModelT]) -> list[ModelT]:
        objs = list(objs)
        self.session.add_all(objs)
        self.session.flush()
        return objs

    def update(self, obj: ModelT, **fields: Any) -> ModelT:
        for k, v in fields.items():
            if not hasattr(obj, k):
                raise ValidationError(f"Unknown field: {k}")
            setattr(obj, k, v)
        self.session.flush()
        return obj

    def delete(self, obj: ModelT) -> None:
        self.session.delete(obj)
        self.session.flush()

    def exists(self, **filters: Any) -> bool:
        return self.get_by(**filters) is not None

    # ---------------- Bulk ----------------
    def bulk_insert(self, rows: list[dict[str, Any]], *, chunk_size: int = 1000) -> int:
        """Fast path for bulk inserts (bypasses ORM per-row overhead)."""
        if not rows:
            return 0
        total = 0
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            self.session.bulk_insert_mappings(self.model, chunk)  # type: ignore[arg-type]
            total += len(chunk)
        self.session.flush()
        return total

    def upsert_by_unique(
        self,
        unique_filter: dict[str, Any],
        values: dict[str, Any],
    ) -> ModelT:
        """Insert if missing, otherwise update. Raises ConflictError if unique filter is empty."""
        if not unique_filter:
            raise ConflictError("upsert_by_unique requires a non-empty unique_filter")
        existing = self.get_by(**unique_filter)
        if existing is None:
            merged = {**unique_filter, **values}
            obj = self.model(**merged)  # type: ignore[call-arg]
            return self.add(obj)
        return self.update(existing, **values)