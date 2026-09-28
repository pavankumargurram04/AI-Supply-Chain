"""Quality-gated promotion.

A model can only be promoted to PRODUCTION when:

    1) It has status STAGING (i.e., it was explicitly moved to staging).
    2) All configured MetricThresholds pass on its test metrics.
    3) If there is a current production model, the candidate must be at least
       `min_relative_improvement` better on the primary metric than the
       incumbent (default 0% — i.e., must be strictly better).

Every promotion attempt is recorded in `model_registry.extras.promotion_history`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from model_registry.registry import Registry, RegistryError
from model_registry.validation import (
    MetricThreshold,
    check_thresholds,
    _get_path,
)

log = get_logger(__name__)


@dataclass
class PromotionCriteria:
    """Thresholds + comparison rules for promotion."""

    # Explicit numeric thresholds on metrics.
    thresholds: List[MetricThreshold] = field(default_factory=list)

    # Primary metric for beating the incumbent.
    # Path is dotted into the metrics dict (e.g. "test.wape").
    primary_metric_path: Optional[str] = "test.wape"
    # Direction: "min" = lower is better, "max" = higher is better.
    primary_metric_direction: str = "min"
    # Relative improvement required (0.0 = strictly better).
    min_relative_improvement: float = 0.0

    def to_dict(self) -> dict:
        return {
            "thresholds": [t.to_dict() for t in self.thresholds],
            "primary_metric_path": self.primary_metric_path,
            "primary_metric_direction": self.primary_metric_direction,
            "min_relative_improvement": self.min_relative_improvement,
        }


@dataclass
class PromotionResult:
    ok: bool
    model_name: str
    version: str
    reason: str = ""
    failures: List[str] = field(default_factory=list)
    comparison: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "model_name": self.model_name,
            "version": self.version,
            "reason": self.reason,
            "failures": list(self.failures),
            "comparison": self.comparison,
        }


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------
def evaluate_promotion(
    registry: Registry,
    *,
    model_name: str,
    version: str,
    criteria: PromotionCriteria,
) -> PromotionResult:
    """Evaluate without mutating. Returns PromotionResult."""
    candidate = registry.get_or_raise(model_name=model_name, version=version)
    failures: List[str] = []

    if candidate.status not in ("staging", "validated"):
        failures.append(
            f"candidate status is {candidate.status!r}; must be 'staging' or 'validated'"
        )

    # Threshold checks.
    metrics = candidate.metrics or {}
    threshold_failures = check_thresholds(metrics, criteria.thresholds)
    failures.extend(threshold_failures)

    # Incumbent comparison.
    comparison: Optional[Dict[str, Any]] = None
    incumbent = registry.get_production(model_name)
    if incumbent is not None and incumbent.id != candidate.id and criteria.primary_metric_path:
        cmp_ = registry.compare(
            model_name=model_name,
            version_a=incumbent.version,
            version_b=candidate.version,
        )
        comparison = cmp_

        inc_val = _get_path(incumbent.metrics or {}, criteria.primary_metric_path)
        cand_val = _get_path(candidate.metrics or {}, criteria.primary_metric_path)

        if inc_val is None or cand_val is None:
            failures.append(
                f"primary metric {criteria.primary_metric_path!r} missing on candidate or incumbent"
            )
        else:
            inc_v = float(inc_val)
            cand_v = float(cand_val)
            if criteria.primary_metric_direction == "min":
                # Lower is better.
                if inc_v > 0:
                    rel = (inc_v - cand_v) / inc_v
                else:
                    rel = 0.0 if cand_v < inc_v else -1.0
            else:
                # Higher is better.
                if inc_v > 0:
                    rel = (cand_v - inc_v) / inc_v
                else:
                    rel = 1.0 if cand_v > inc_v else 0.0
            if rel < criteria.min_relative_improvement:
                failures.append(
                    f"candidate {criteria.primary_metric_path}={cand_v:.4f} "
                    f"vs incumbent {inc_v:.4f}: relative improvement {rel:.2%} "
                    f"< required {criteria.min_relative_improvement:.2%}"
                )

    if failures:
        return PromotionResult(
            ok=False, model_name=model_name, version=version,
            reason="promotion criteria not satisfied",
            failures=failures, comparison=comparison,
        )

    return PromotionResult(
        ok=True, model_name=model_name, version=version,
        reason="all criteria satisfied",
        failures=[], comparison=comparison,
    )


def promote(
    registry: Registry,
    *,
    model_name: str,
    version: str,
    criteria: PromotionCriteria,
    actor: Optional[str] = None,
) -> PromotionResult:
    """Evaluate + execute promotion. Records the decision in extras.promotion_history."""
    result = evaluate_promotion(
        registry, model_name=model_name, version=version, criteria=criteria,
    )

    row = registry.get_or_raise(model_name=model_name, version=version)
    history = list((row.extras or {}).get("promotion_history") or [])
    history.append({
        "at": datetime.now(tz=timezone.utc).isoformat(),
        "actor": actor,
        "criteria": criteria.to_dict(),
        "result": result.to_dict(),
    })
    extras = dict(row.extras or {})
    extras["promotion_history"] = history
    row.extras = extras
    registry.session.flush()

    if not result.ok:
        log.info("registry.promotion.rejected", extra={
            "model_name": model_name, "version": version,
            "failures": result.failures,
        })
        return result

    registry.mark_production(model_name=model_name, version=version)
    log.info("registry.promotion.ok", extra={
        "model_name": model_name, "version": version,
    })
    return result