"""Model registry: versioning, artifact storage, promotion, rollback.

Public API:
    from model_registry import Registry, ModelVersionSpec, PromotionCriteria

    spec = ModelVersionSpec(
        model_name="demand_lstm_global",
        model_family="demand_lstm",
        artifact_dir="artifacts/demand/dev",
        dataset_version="ds_v1",
        feature_version="v1_abc123",
        hyperparameters={...},
        metrics={"val": {...}, "test": {...}},
        framework="pytorch",
        trained_by="scripts.train_all",
    )
    reg = Registry(session)
    version = reg.register(spec)
    reg.promote(model_name="demand_lstm_global", version=version, criteria=PromotionCriteria(...))
"""

from model_registry.metadata import ModelCard, build_model_card
from model_registry.promotion import PromotionCriteria, PromotionResult
from model_registry.registry import (
    ModelVersionSpec,
    Registry,
    RegistryError,
)
from model_registry.versioning import (
    compute_dataset_version,
    compute_feature_version_from_registry,
    compute_model_version_id,
    bump_semver,
)

__all__ = [
    "Registry",
    "RegistryError",
    "ModelVersionSpec",
    "PromotionCriteria",
    "PromotionResult",
    "ModelCard",
    "build_model_card",
    "compute_dataset_version",
    "compute_feature_version_from_registry",
    "compute_model_version_id",
    "bump_semver",
]