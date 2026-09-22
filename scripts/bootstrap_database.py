"""Create database (if missing), run migrations, seed admin user.

Idempotent: safe to run multiple times.

Usage:
    python scripts/bootstrap_database.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine.url import make_url  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.database import check_database, session_scope  # noqa: E402
from core.enums import UserRole  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402
from core.security import hash_password  # noqa: E402
from db.repositories.user_repository import UserRepository  # noqa: E402


log = get_logger("bootstrap")


def _ensure_database_exists() -> None:
    """Create the configured database if it does not yet exist (PostgreSQL)."""
    settings = get_settings()
    url = make_url(settings.database_url)
    target_db = url.database

    admin_url = url.set(database="postgres")
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    try:
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"),
                {"n": target_db},
            ).scalar_one_or_none()
            if exists:
                log.info("db.exists", extra={"db": target_db})
                return
            # Identifier must be quoted manually since bind params are not
            # allowed for DDL identifiers.
            safe = target_db.replace('"', '""')
            conn.execute(text(f'CREATE DATABASE "{safe}"'))
            log.info("db.created", extra={"db": target_db})
    finally:
        engine.dispose()


def _run_migrations() -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "db" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    command.upgrade(cfg, "head")
    log.info("migrations.applied")


def _seed_admin() -> None:
    settings = get_settings()
    with session_scope() as session:
        repo = UserRepository(session)
        if repo.get_by_username("admin") is None:
            repo.create(
                email="admin@example.com",
                username="admin",
                hashed_password=hash_password("admin"),
                role=UserRole.ADMIN.value,
                full_name="Platform Administrator",
            )
            log.info("user.seeded", extra={"username": "admin"})
        else:
            log.info("user.exists", extra={"username": "admin"})


def main() -> int:
    configure_logging()
    log.info("bootstrap.start", extra={"env": get_settings().APP_ENV})

    _ensure_database_exists()
    _run_migrations()

    if not check_database():
        log.error("db.connectivity_failed")
        return 1

    _seed_admin()
    log.info("bootstrap.done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())