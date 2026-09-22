"""MinIO / S3-compatible object store wrapper.

Buckets are created on demand. Clients are cached per process.
"""

from __future__ import annotations

import io
import os
from typing import Optional

from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)

_client = None
_ensured_buckets: set[str] = set()


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from minio import Minio  # type: ignore
        settings = get_settings()
        _client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        log.info("object_store.connected", extra={"endpoint": settings.MINIO_ENDPOINT})
    except Exception as exc:  # pragma: no cover
        log.warning("object_store.unavailable", extra={"error": str(exc)})
        _client = None
    return _client


def _ensure_bucket(client, bucket: str) -> None:
    if bucket in _ensured_buckets:
        return
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        _ensured_buckets.add(bucket)
    except Exception as exc:  # pragma: no cover
        log.warning("object_store.bucket_check_failed", extra={"bucket": bucket, "error": str(exc)})


def put_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
    client = _get_client()
    if client is None:
        return False
    _ensure_bucket(client, bucket)
    client.put_object(bucket, key, io.BytesIO(data), length=len(data), content_type=content_type)
    return True


def put_file(bucket: str, key: str, path: str) -> bool:
    client = _get_client()
    if client is None:
        return False
    _ensure_bucket(client, bucket)
    client.fput_object(bucket, key, path)
    return True


def get_bytes(bucket: str, key: str) -> Optional[bytes]:
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.get_object(bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()
    except Exception:
        return None


def list_objects(bucket: str, prefix: str = "") -> list[str]:
    client = _get_client()
    if client is None:
        return []
    try:
        return [obj.object_name for obj in client.list_objects(bucket, prefix=prefix, recursive=True)]
    except Exception:
        return []


def ensure_default_buckets() -> None:
    client = _get_client()
    if client is None:
        return
    settings = get_settings()
    for b in (settings.MINIO_BUCKET_MODELS, settings.MINIO_BUCKET_DATA, settings.MINIO_BUCKET_REPORTS):
        _ensure_bucket(client, b)


def local_dir_for(kind: str) -> str:
    """Return (and create) a local fallback directory for a given kind."""
    settings = get_settings()
    mapping = {
        "models": settings.ARTIFACTS_DIR,
        "data": settings.DATASETS_DIR,
        "reports": settings.REPORTS_DIR,
    }
    path = mapping.get(kind, settings.ARTIFACTS_DIR)
    os.makedirs(path, exist_ok=True)
    return path