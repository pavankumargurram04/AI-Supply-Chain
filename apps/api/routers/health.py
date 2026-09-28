"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from core.cache import cache_ping
from core.config import get_settings
from core.database import check_database

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness — always returns 200 if the process is up."""
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.APP_ENV,
    }


@router.get("/readyz")
def readiness() -> dict:
    """Readiness — checks DB + cache."""
    db_ok = False
    db_err = None
    try:
        db_ok = check_database()
    except Exception as e:  # noqa: BLE001
        db_err = str(e)

    cache_ok = cache_ping()

    return {
        "status": "ready" if (db_ok and cache_ok) else "degraded",
        "checks": {
            "database": {"ok": db_ok, "error": db_err},
            "cache": {"ok": cache_ok},
        },
    }