"""SQLAlchemy 2.0 engine/session wiring.

- `engine` — module-level engine built lazily from settings.
- `SessionLocal` — sessionmaker bound to the engine.
- `get_db()` — FastAPI dependency yielding a session.
- `session_scope()` — context manager for scripts and Celery tasks.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from core.config import get_settings
from core.errors import DatabaseError
from core.logging import get_logger

log = get_logger(__name__)


_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def _build_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        echo=settings.DB_ECHO,
        future=True,
    )


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
        log.info("db.engine.created", extra={"url": _mask_url(get_settings().database_url)})
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
    return _SessionLocal


def _mask_url(url: str) -> str:
    # Hide password in logs.
    if "@" not in url:
        return url
    scheme_userinfo, hostpart = url.split("@", 1)
    if "://" in scheme_userinfo:
        scheme, _ = scheme_userinfo.split("://", 1)
        return f"{scheme}://***@{hostpart}"
    return "***@" + hostpart


# ---------------- FastAPI dependency ----------------
def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, committed on success.

    On any exception during the request, the transaction is rolled back
    before the session is closed. This makes every write endpoint atomic.
    """
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------- Scripts / Celery ----------------
@contextlib.contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session scope. Commits on success, rolls back on error."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        log.exception("db.session.rollback", extra={"error": str(exc)})
        raise
    finally:
        session.close()


def check_database() -> bool:
    """Lightweight connectivity check. Returns True on success."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        raise DatabaseError(f"Database connectivity check failed: {exc}") from exc


def dispose_engine() -> None:
    """Dispose the engine (used in tests / shutdown)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None