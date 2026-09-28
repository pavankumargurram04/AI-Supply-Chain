"""Registry operations on the `model_registry` table.

Every model version is one row. Content-addressed version ids make repeated
registration of the same inputs a no-op — the registry returns the existing
row id and does not create a duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from core.enums import ModelStatus
from core.logging import get_logger
from db.models.model_registry import ModelRegistry
from model_registry.artifact_store import ArtifactLocation, ArtifactStore
from model_registry.metadata import ModelCard, build_model_card
from model_registry.validation import validate_spec
from model_registry.versioning import compute_model_version_id

log = get_logger(__name__)


class RegistryError(RuntimeError):
    """Raised when a registry operation cannot proceed."""


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass
class ModelVersionSpec:
    model_name: str
    model_family: str
    framework: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    dataset_version: Optional[str] = None
    feature_version: Optional[str] = None
    artifact_dir: Optional[str] = None       # local path to upload
    trained_by: str = "scripts.train_all"
    training_data_summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "framework": self.framework,
            "hyperparameters": dict(self.hyperparameters),
            "metrics": dict(self.metrics),
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "artifact_dir": self.artifact_dir,
            "trained_by": self.trained_by,
            "training_data_summary": dict(self.training_data_summary),
            "notes": list(self.notes),
            "extras": dict(self.extras),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class Registry:
    def __init__(self, session: Session, *, artifact_store: Optional[ArtifactStore] = None) -> None:
        self.session = session
        self.store = artifact_store or ArtifactStore()

    # ---------------- Registration ------------------------------------
    def register(self, spec: ModelVersionSpec) -> ModelRegistry:
        """Register a new model version.

        If a row with the same content-addressed version already exists, it
        is returned unchanged (no duplicate).
        """
        validate_spec(spec.to_dict())

        version_id = compute_model_version_id(
            model_name=spec.model_name,
            model_family=spec.model_family,
            dataset_version=spec.dataset_version,
            feature_version=spec.feature_version,
            hyperparameters=spec.hyperparameters,
        )

        existing = (
            self.session.query(ModelRegistry)
            .filter_by(model_name=spec.model_name, version=version_id)
            .one_or_none()
        )
        if existing is not None:
            log.info("registry.register.noop", extra={
                "model_name": spec.model_name, "version": version_id,
            })
            return existing

        # Upload artifacts.
        artifact_uri: Optional[str] = None
        artifact_path: Optional[str] = None
        if spec.artifact_dir is not None:
            loc = self.store.upload_directory(
                model_name=spec.model_name,
                version=version_id,
                src_dir=spec.artifact_dir,
            )
            artifact_uri = loc.uri
            artifact_path = spec.artifact_dir

        # Build model card.
        card: ModelCard = build_model_card(
            model_name=spec.model_name,
            model_family=spec.model_family,
            version=version_id,
            dataset_version=spec.dataset_version,
            feature_version=spec.feature_version,
            framework=spec.framework,
            hyperparameters=spec.hyperparameters,
            metrics=spec.metrics,
            trained_by=spec.trained_by,
            training_data_summary=spec.training_data_summary,
            notes=spec.notes,
        )
        # Upload the card as a file next to the artifact.
        try:
            card_bytes = card.to_dict().__repr__().encode("utf-8")
            self.store.put_single_file(
                model_name=spec.model_name, version=version_id,
                filename="model_card.json",
                data=__import__("json").dumps(card.to_dict(), indent=2, default=str).encode("utf-8"),
                content_type="application/json",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("registry.card_upload_failed", extra={"error": str(e)})

        row = ModelRegistry(
            model_name=spec.model_name,
            model_family=spec.model_family,
            version=version_id,
            dataset_version=spec.dataset_version,
            feature_version=spec.feature_version,
            framework=spec.framework,
            artifact_path=artifact_path,
            artifact_uri=artifact_uri,
            hyperparameters=spec.hyperparameters,
            metrics=spec.metrics,
            extras={**(spec.extras or {}), "model_card": card.to_dict()},
            status=ModelStatus.TRAINED.value,
            trained_at=datetime.now(tz=timezone.utc),
        )
        self.session.add(row)
        self.session.flush()
        log.info("registry.registered", extra={
            "model_name": spec.model_name, "version": version_id,
            "artifact_uri": artifact_uri,
        })
        return row

    # ---------------- Lookups -----------------------------------------
    def get(self, *, model_name: str, version: str) -> Optional[ModelRegistry]:
        return (
            self.session.query(ModelRegistry)
            .filter_by(model_name=model_name, version=version)
            .one_or_none()
        )

    def get_or_raise(self, *, model_name: str, version: str) -> ModelRegistry:
        row = self.get(model_name=model_name, version=version)
        if row is None:
            raise RegistryError(f"Model not found: {model_name} {version}")
        return row

    def list_for_family(self, model_family: str) -> List[ModelRegistry]:
        return (
            self.session.query(ModelRegistry)
            .filter_by(model_family=model_family)
            .order_by(ModelRegistry.created_at.desc())
            .all()
        )

    def list_by_status(self, status: str) -> List[ModelRegistry]:
        return (
            self.session.query(ModelRegistry)
            .filter_by(status=status)
            .order_by(ModelRegistry.created_at.desc())
            .all()
        )

    def get_production(self, model_name: str) -> Optional[ModelRegistry]:
        return (
            self.session.query(ModelRegistry)
            .filter_by(model_name=model_name, status=ModelStatus.PRODUCTION.value)
            .order_by(ModelRegistry.promoted_at.desc())
            .first()
        )

    # ---------------- Status transitions ------------------------------
    def mark_validated(self, *, model_name: str, version: str) -> ModelRegistry:
        row = self.get_or_raise(model_name=model_name, version=version)
        if row.status not in (ModelStatus.TRAINED.value,):
            raise RegistryError(
                f"Cannot mark VALIDATED from status {row.status!r}"
            )
        row.status = ModelStatus.VALIDATED.value
        row.validated_at = datetime.now(tz=timezone.utc)
        self.session.flush()
        return row

    def mark_staging(self, *, model_name: str, version: str) -> ModelRegistry:
        row = self.get_or_raise(model_name=model_name, version=version)
        if row.status not in (ModelStatus.VALIDATED.value, ModelStatus.STAGING.value):
            raise RegistryError(
                f"Cannot mark STAGING from status {row.status!r}"
            )
        row.status = ModelStatus.STAGING.value
        self.session.flush()
        return row

    def mark_production(self, *, model_name: str, version: str) -> ModelRegistry:
        """Promote to production. Any current production model is archived first."""
        row = self.get_or_raise(model_name=model_name, version=version)
        if row.status not in (ModelStatus.STAGING.value, ModelStatus.VALIDATED.value):
            raise RegistryError(
                f"Cannot promote to PRODUCTION from status {row.status!r}"
            )

        current = self.get_production(model_name)
        if current is not None and current.id != row.id:
            current.status = ModelStatus.ARCHIVED.value
            current.archived_at = datetime.now(tz=timezone.utc)
            log.info("registry.archived_previous", extra={
                "model_name": model_name, "version": current.version,
            })

        row.status = ModelStatus.PRODUCTION.value
        row.promoted_at = datetime.now(tz=timezone.utc)
        self.session.flush()
        log.info("registry.promoted", extra={
            "model_name": model_name, "version": version,
        })
        return row

    def mark_archived(self, *, model_name: str, version: str) -> ModelRegistry:
        row = self.get_or_raise(model_name=model_name, version=version)
        if row.status == ModelStatus.PRODUCTION.value:
            raise RegistryError(
                "Refusing to archive the current PRODUCTION model. "
                "Promote a successor or run rollback instead."
            )
        row.status = ModelStatus.ARCHIVED.value
        row.archived_at = datetime.now(tz=timezone.utc)
        self.session.flush()
        return row

    def mark_rejected(
        self, *, model_name: str, version: str, reason: str
    ) -> ModelRegistry:
        row = self.get_or_raise(model_name=model_name, version=version)
        row.status = ModelStatus.REJECTED.value
        extras = dict(row.extras or {})
        extras["rejected_reason"] = reason
        extras["rejected_at"] = datetime.now(tz=timezone.utc).isoformat()
        row.extras = extras
        self.session.flush()
        return row

    # ---------------- Compare -----------------------------------------
    def compare(self, *, model_name: str, version_a: str, version_b: str) -> Dict[str, Any]:
        a = self.get_or_raise(model_name=model_name, version=version_a)
        b = self.get_or_raise(model_name=model_name, version=version_b)
        return {
            "model_name": model_name,
            "a": {"version": a.version, "metrics": a.metrics,
                  "hyperparameters": a.hyperparameters, "status": a.status},
            "b": {"version": b.version, "metrics": b.metrics,
                  "hyperparameters": b.hyperparameters, "status": b.status},
            "metric_deltas": _metric_deltas(a.metrics or {}, b.metrics or {}),
        }


def _metric_deltas(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Compute b − a for every numeric metric present in both."""
    out: Dict[str, Any] = {}
    for k, va in a.items():
        if isinstance(va, dict) and isinstance(b.get(k), dict):
            sub = _metric_deltas(va, b[k])
            if sub:
                out[k] = sub
        elif isinstance(va, (int, float)) and isinstance(b.get(k), (int, float)):
            out[k] = float(b[k]) - float(va)
    return out