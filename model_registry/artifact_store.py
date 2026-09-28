"""Artifact storage: MinIO primary, local directory fallback.

Public API:
    store = ArtifactStore()
    uri = store.upload_directory(model_name="demand_lstm_global", version="mv_abc", src_dir="artifacts/demand/dev")
    local_dir = store.download_directory(model_name=..., version=..., dest_dir="tmp/downloaded")

URIs are of the form:
    s3://<bucket>/<model_name>/<version>/           when MinIO is reachable
    file://<abs-path>/<model_name>/<version>/       when using local fallback
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.logging import get_logger
from core.object_store import (
    ensure_default_buckets,
    get_bytes,
    list_objects,
    local_dir_for,
    put_bytes,
    put_file,
)

log = get_logger(__name__)


@dataclass
class ArtifactLocation:
    uri: str
    kind: str          # "s3" | "file"
    bucket: Optional[str]
    prefix: str


class ArtifactStore:
    def __init__(self, *, model_bucket: Optional[str] = None) -> None:
        from core.config import get_settings
        s = get_settings()
        self.model_bucket = model_bucket or s.MINIO_BUCKET_MODELS
        self.local_root = Path(local_dir_for("models"))
        self._s3_available: Optional[bool] = None

    # ------------------------------------------------------------------
    def _check_s3(self) -> bool:
        if self._s3_available is not None:
            return self._s3_available
        try:
            ensure_default_buckets()
            # A no-op list confirms connectivity.
            list_objects(self.model_bucket, prefix="_ping")
            self._s3_available = True
        except Exception as e:  # noqa: BLE001
            log.warning("artifact_store.s3_unavailable", extra={"error": str(e)})
            self._s3_available = False
        return self._s3_available

    # ------------------------------------------------------------------
    def upload_directory(
        self,
        *,
        model_name: str,
        version: str,
        src_dir: str | Path,
    ) -> ArtifactLocation:
        """Upload every file in `src_dir` under `model_name/version/`.

        Returns the URI + kind so the caller can persist it.
        """
        src = Path(src_dir)
        if not src.exists():
            raise FileNotFoundError(f"Artifact directory not found: {src}")

        prefix = f"{model_name}/{version}"

        if self._check_s3():
            for p in sorted(src.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(src).as_posix()
                    key = f"{prefix}/{rel}"
                    try:
                        put_file(self.model_bucket, key, str(p))
                    except Exception as e:  # noqa: BLE001
                        log.warning("artifact_store.upload_failed",
                                    extra={"file": str(p), "error": str(e)})
                        # Fall back per-file to local storage for the whole dir.
                        return self._upload_local(src, prefix)
            return ArtifactLocation(
                uri=f"s3://{self.model_bucket}/{prefix}/",
                kind="s3",
                bucket=self.model_bucket,
                prefix=prefix,
            )

        return self._upload_local(src, prefix)

    def _upload_local(self, src: Path, prefix: str) -> ArtifactLocation:
        dest = self.local_root / prefix
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        uri = f"file://{dest.as_posix()}/"
        log.info("artifact_store.upload_local", extra={"uri": uri, "files": len(list(dest.rglob('*')))})
        return ArtifactLocation(uri=uri, kind="file", bucket=None, prefix=prefix)

    # ------------------------------------------------------------------
    def download_directory(
        self,
        *,
        uri: str,
        dest_dir: str | Path,
    ) -> Path:
        """Restore an artifact directory from a URI (s3 or file)."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        if uri.startswith("file://"):
            src = Path(uri.replace("file://", "").rstrip("/"))
            if not src.exists():
                raise FileNotFoundError(f"Local artifact path missing: {src}")
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            return dest

        if uri.startswith("s3://"):
            # s3://bucket/prefix/... → bucket=..., prefix=...
            rest = uri.replace("s3://", "").rstrip("/")
            bucket, _, prefix = rest.partition("/")
            keys = list_objects(bucket, prefix=prefix + "/")
            if not keys:
                raise FileNotFoundError(f"No objects under {uri}")
            for key in keys:
                data = get_bytes(bucket, key)
                if data is None:
                    continue
                rel = key[len(prefix) + 1 :]
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            return dest

        raise ValueError(f"Unsupported artifact URI scheme: {uri}")

    # ------------------------------------------------------------------
    def put_single_file(
        self,
        *,
        model_name: str,
        version: str,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> ArtifactLocation:
        prefix = f"{model_name}/{version}"
        key = f"{prefix}/{filename}"
        if self._check_s3():
            put_bytes(self.model_bucket, key, data, content_type=content_type)
            return ArtifactLocation(
                uri=f"s3://{self.model_bucket}/{key}",
                kind="s3",
                bucket=self.model_bucket,
                prefix=key,
            )
        dest = self.local_root / prefix
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / filename
        target.write_bytes(data)
        return ArtifactLocation(
            uri=f"file://{target.as_posix()}",
            kind="file",
            bucket=None,
            prefix=key,
        )