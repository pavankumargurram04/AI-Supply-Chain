"""Deterministic versioning helpers.

Three independent content-addressed version identifiers:

- dataset_version: SHA-256 of (mode, seed, rows-per-table, generated_at).
- feature_version: from `feature_registry.FEATURE_VERSION` (already
  content-addressed) or overridable by the caller.
- model_version_id: SHA-256 of (model_name, feature_version, dataset_version,
  hyperparameters canonical form). This makes re-training with identical
  inputs produce the same version id — which the registry treats as a no-op.

`bump_semver` handles human-facing semantic version strings ("v1.2.3").
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Content-addressed IDs
# ---------------------------------------------------------------------------
def compute_dataset_version(
    *,
    mode: str,
    seed: int,
    row_counts: Dict[str, int],
    generated_at: Optional[str] = None,
) -> str:
    """Dataset version is content-addressed from its defining metadata.

    `generated_at` is optional. When provided it is included; when omitted
    the version is purely structural (safe for tests).
    """
    payload = {
        "mode": mode,
        "seed": int(seed),
        "row_counts": {k: int(v) for k, v in sorted(row_counts.items())},
    }
    if generated_at is not None:
        payload["generated_at"] = generated_at
    return "ds_" + _hash(_canonical(payload))[:12]


def compute_feature_version_from_registry() -> str:
    """Delegates to the feature registry's own content-addressed version."""
    from data_platform.features.feature_registry import FEATURE_VERSION
    return FEATURE_VERSION


def compute_model_version_id(
    *,
    model_name: str,
    model_family: str,
    dataset_version: Optional[str],
    feature_version: Optional[str],
    hyperparameters: Dict[str, Any],
) -> str:
    """Content-addressed model id. Same inputs → same id → registry treats as no-op."""
    payload = {
        "model_name": model_name,
        "model_family": model_family,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "hyperparameters": hyperparameters,
    }
    return "mv_" + _hash(_canonical(payload))[:16]


# ---------------------------------------------------------------------------
# Semantic version strings (human-facing)
# ---------------------------------------------------------------------------
def bump_semver(current: Optional[str], kind: str = "patch") -> str:
    """Return the next semver string after `current`, using `kind` bump.

    Accepted formats: "v1", "v1.2", "v1.2.3", or None (starts at "v0.1.0").
    """
    if current is None:
        return "v0.1.0"
    s = current.lstrip("v")
    parts = s.split(".")
    while len(parts) < 3:
        parts.append("0")
    try:
        major, minor, patch = (int(p) for p in parts[:3])
    except ValueError as e:
        raise ValueError(f"Unparseable semver: {current!r}") from e

    if kind == "major":
        return f"v{major + 1}.0.0"
    if kind == "minor":
        return f"v{major}.{minor + 1}.0"
    if kind == "patch":
        return f"v{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unknown bump kind: {kind}")


# ---------------------------------------------------------------------------
# Directory checksums (for artifact integrity)
# ---------------------------------------------------------------------------
def directory_digest(path: str | Path) -> str:
    """SHA-256 of all file names + contents under `path` (deterministic order).

    Used to detect whether an artifact directory has changed between two
    registry entries with the same hyperparameters.
    """
    root = Path(path)
    if not root.exists():
        return "missing"
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\x00")
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            h.update(b"\x01")
    return "art_" + h.hexdigest()[:16]