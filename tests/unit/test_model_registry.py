"""Model registry tests.

Uses a real Postgres test database (skipped if unavailable). Exercises:
    - content-addressed versioning
    - registration idempotence
    - status transition rules
    - quality-gated promotion
    - atomic rollback
"""

from __future__ import annotations

import pytest

from core.enums import ModelStatus
from model_registry.registry import ModelVersionSpec, Registry, RegistryError
from model_registry.promotion import (
    MetricThreshold,
    PromotionCriteria,
    evaluate_promotion,
    promote,
)
from model_registry.rollback import rollback
from model_registry.validation import validate_spec
from model_registry.versioning import (
    bump_semver,
    compute_dataset_version,
    compute_model_version_id,
)


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Versioning (pure) — runs without DB.
# ---------------------------------------------------------------------------
def test_compute_dataset_version_is_deterministic():
    a = compute_dataset_version(mode="dev", seed=42,
                                row_counts={"demand_records": 1000})
    b = compute_dataset_version(mode="dev", seed=42,
                                row_counts={"demand_records": 1000})
    c = compute_dataset_version(mode="dev", seed=43,
                                row_counts={"demand_records": 1000})
    assert a == b
    assert a != c


def test_compute_model_version_id_changes_on_hyperparameters():
    a = compute_model_version_id(
        model_name="m", model_family="demand_lstm",
        dataset_version="ds_a", feature_version="v1",
        hyperparameters={"lr": 0.001},
    )
    b = compute_model_version_id(
        model_name="m", model_family="demand_lstm",
        dataset_version="ds_a", feature_version="v1",
        hyperparameters={"lr": 0.002},
    )
    assert a != b


def test_bump_semver():
    assert bump_semver(None) == "v0.1.0"
    assert bump_semver("v1.2.3", "patch") == "v1.2.4"
    assert bump_semver("v1.2.3", "minor") == "v1.3.0"
    assert bump_semver("v1.2.3", "major") == "v2.0.0"


# ---------------------------------------------------------------------------
# Spec validation (pure)
# ---------------------------------------------------------------------------
def test_validate_spec_rejects_unknown_family():
    with pytest.raises(Exception):
        validate_spec({
            "model_name": "x", "model_family": "unknown",
            "framework": "pytorch", "hyperparameters": {}, "metrics": {},
        })


def test_metric_threshold_max_and_min():
    th = MetricThreshold(metric_path="test.wape", max_value=0.25)
    assert th.check({"test": {"wape": 0.20}}) is None
    assert th.check({"test": {"wape": 0.30}}) is not None

    th2 = MetricThreshold(metric_path="test.roc_auc", min_value=0.75)
    assert th2.check({"test": {"roc_auc": 0.80}}) is None
    assert th2.check({"test": {"roc_auc": 0.70}}) is not None


# ---------------------------------------------------------------------------
# Registry (needs DB)
# ---------------------------------------------------------------------------
def _spec(name="demand_lstm_global", family="demand_lstm", test_wape=0.20, lr=1e-3):
    return ModelVersionSpec(
        model_name=name,
        model_family=family,
        framework="pytorch",
        hyperparameters={"lr": lr, "epochs": 5},
        metrics={
            "val": {"wape": test_wape + 0.02},
            "test": {"wape": test_wape},
        },
        dataset_version="ds_dev",
        feature_version="v1_abc",
        artifact_dir=None,
        trained_by="test",
    )


def test_registry_register_is_idempotent(db_session):
    reg = Registry(db_session)
    spec = _spec()
    a = reg.register(spec)
    b = reg.register(_spec())
    assert a.id == b.id


def test_registry_status_transitions_enforced(db_session):
    reg = Registry(db_session)
    row = reg.register(_spec(lr=0.0007))

    with pytest.raises(RegistryError):
        reg.mark_staging(model_name=row.model_name, version=row.version)
    # Correct path:
    reg.mark_validated(model_name=row.model_name, version=row.version)
    reg.mark_staging(model_name=row.model_name, version=row.version)
    reg.mark_production(model_name=row.model_name, version=row.version)
    assert reg.get_production(row.model_name).version == row.version


def test_promotion_rejects_when_thresholds_fail(db_session):
    reg = Registry(db_session)
    row = reg.register(_spec(test_wape=0.40, lr=0.0004))   # bad wape
    reg.mark_validated(model_name=row.model_name, version=row.version)
    reg.mark_staging(model_name=row.model_name, version=row.version)

    criteria = PromotionCriteria(
        thresholds=[MetricThreshold(metric_path="test.wape", max_value=0.25)],
        primary_metric_path=None,
    )
    result = evaluate_promotion(reg, model_name=row.model_name,
                                version=row.version, criteria=criteria)
    assert not result.ok
    assert any("exceeds max" in f for f in result.failures)


def test_promotion_beats_incumbent(db_session):
    reg = Registry(db_session)
    # Incumbent v1
    v1 = reg.register(_spec(test_wape=0.20, lr=0.0001))
    reg.mark_validated(model_name=v1.model_name, version=v1.version)
    reg.mark_staging(model_name=v1.model_name, version=v1.version)
    reg.mark_production(model_name=v1.model_name, version=v1.version)

    # Candidate v2 — better wape
    v2 = reg.register(_spec(test_wape=0.15, lr=0.0002))
    reg.mark_validated(model_name=v2.model_name, version=v2.version)
    reg.mark_staging(model_name=v2.model_name, version=v2.version)

    criteria = PromotionCriteria(
        thresholds=[MetricThreshold(metric_path="test.wape", max_value=0.25)],
        primary_metric_path="test.wape",
        primary_metric_direction="min",
        min_relative_improvement=0.0,
    )
    result = promote(reg, model_name=v2.model_name, version=v2.version, criteria=criteria)
    assert result.ok, result.failures
    assert reg.get_production(v2.model_name).version == v2.version
    # v1 must be archived.
    v1_row = reg.get(model_name=v1.model_name, version=v1.version)
    assert v1_row.status == ModelStatus.ARCHIVED.value


def test_promotion_rejects_worse_than_incumbent(db_session):
    reg = Registry(db_session)
    v1 = reg.register(_spec(test_wape=0.15, lr=0.001))
    reg.mark_validated(model_name=v1.model_name, version=v1.version)
    reg.mark_staging(model_name=v1.model_name, version=v1.version)
    reg.mark_production(model_name=v1.model_name, version=v1.version)

    v2 = reg.register(_spec(test_wape=0.20, lr=0.002))   # worse
    reg.mark_validated(model_name=v2.model_name, version=v2.version)
    reg.mark_staging(model_name=v2.model_name, version=v2.version)

    criteria = PromotionCriteria(
        thresholds=[MetricThreshold(metric_path="test.wape", max_value=0.25)],
        primary_metric_path="test.wape",
        primary_metric_direction="min",
        min_relative_improvement=0.0,
    )
    result = promote(reg, model_name=v2.model_name, version=v2.version, criteria=criteria)
    assert not result.ok
    # Production must remain v1.
    assert reg.get_production(v1.model_name).version == v1.version


def test_rollback_swaps_production_to_archived(db_session):
    reg = Registry(db_session)
    v1 = reg.register(_spec(test_wape=0.15, lr=0.001))
    reg.mark_validated(model_name=v1.model_name, version=v1.version)
    reg.mark_staging(model_name=v1.model_name, version=v1.version)
    reg.mark_production(model_name=v1.model_name, version=v1.version)

    v2 = reg.register(_spec(test_wape=0.10, lr=0.002))
    reg.mark_validated(model_name=v2.model_name, version=v2.version)
    reg.mark_staging(model_name=v2.model_name, version=v2.version)
    reg.mark_production(model_name=v2.model_name, version=v2.version)

    assert reg.get_production(v1.model_name).version == v2.version

    result = rollback(reg, model_name=v1.model_name, actor="test",
                      reason="regression detected")
    assert result.ok
    assert result.from_version == v2.version
    assert result.to_version == v1.version
    assert reg.get_production(v1.model_name).version == v1.version


def test_rollback_explicit_target(db_session):
    reg = Registry(db_session)
    v1 = reg.register(_spec(test_wape=0.15, lr=0.001))
    reg.mark_validated(model_name=v1.model_name, version=v1.version)
    reg.mark_staging(model_name=v1.model_name, version=v1.version)
    reg.mark_production(model_name=v1.model_name, version=v1.version)

    v2 = reg.register(_spec(test_wape=0.10, lr=0.002))
    reg.mark_validated(model_name=v2.model_name, version=v2.version)
    reg.mark_staging(model_name=v2.model_name, version=v2.version)
    reg.mark_production(model_name=v2.model_name, version=v2.version)

    result = rollback(reg, model_name=v1.model_name,
                      to_version=v1.version, reason="manual")
    assert result.ok
    assert reg.get_production(v1.model_name).version == v1.version


def test_rollback_no_target_returns_failure(db_session):
    reg = Registry(db_session)
    v1 = reg.register(_spec(test_wape=0.15, lr=0.001))
    reg.mark_validated(model_name=v1.model_name, version=v1.version)
    reg.mark_staging(model_name=v1.model_name, version=v1.version)
    reg.mark_production(model_name=v1.model_name, version=v1.version)

    result = rollback(reg, model_name=v1.model_name)
    assert not result.ok
    assert "no archived version" in result.reason.lower()