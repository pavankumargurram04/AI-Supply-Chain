"""Redis-backed cache helper. Optional — degrades gracefully if unavailable."""

from __future__ import annotations

import json
from typing import Any, Optional

from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import redis  # type: ignore
        _client = redis.Redis.from_url(
            get_settings().redis_url, decode_responses=True
        )
        _client.ping()
        log.info("cache.connected")
    except Exception as exc:  # pragma: no cover - env dependent
        log.warning("cache.unavailable", extra={"error": str(exc)})
        _client = None
    return _client


def cache_get(key: str) -> Optional[Any]:
    client = _get_client()
    if client is None:
        return None
    raw = client.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.setex(key, ttl_seconds, json.dumps(value, default=str))
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("cache.set_failed", extra={"error": str(exc)})
        return False


def cache_delete(key: str) -> bool:
    client = _get_client()
    if client is None:
        return False
    return bool(client.delete(key))


def cache_ping() -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except Exception:
        return False