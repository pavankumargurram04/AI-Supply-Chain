"""Validation of registry entries and promotion criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class ValidationError(ValueError):
    pass


_REQUIRED_SPEC_FIELDS = (
    "model_name", "model_family", "framework",
    "hyperparameters", "metrics",
)

_ALLOWED_FAMILIES = {
    "demand_lstm",
    "supplier_xgb",
    "inventory_dqn",
    "procurement_opt",
}


def validate_spec(spec: Dict[str, Any]) -> None:
    for f in _REQUIRED_SPEC_FIELDS:
        if f not in spec or spec[f] in (None, ""):
            raise ValidationError(f"ModelVersionSpec missing required field: {f}")
    if spec["model_family"] not in _ALLOWED_FAMILIES:
        raise ValidationError(
            f"Unknown model_family={spec['model_family']!r}. "
            f"Allowed: {sorted(_ALLOWED_FAMILIES)}"
        )
    if not isinstance(spec["metrics"], dict):
        raise ValidationError("metrics must be a dict")
    if not isinstance(spec["hyperparameters"], dict):
        raise ValidationError("hyperparameters must be a dict")


@dataclass
class MetricThreshold:
    """A single metric threshold used in promotion."""

    metric_path: str          # e.g. "test.wape" — dotted path into the metrics dict
    max_value: Optional[float] = None      # for "lower is better" metrics
    min_value: Optional[float] = None      # for "higher is better" metrics

    def check(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Return None if satisfied, else a human-readable failure message."""
        value = _get_path(metrics, self.metric_path)
        if value is None:
            return f"metric {self.metric_path!r} missing from metrics"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return f"metric {self.metric_path!r} is not numeric ({value!r})"
        if self.max_value is not None and v > self.max_value:
            return f"{self.metric_path}={v:.4f} exceeds max {self.max_value:.4f}"
        if self.min_value is not None and v < self.min_value:
            return f"{self.metric_path}={v:.4f} below min {self.min_value:.4f}"
        return None

    def to_dict(self) -> dict:
        return {"metric_path": self.metric_path,
                "max_value": self.max_value,
                "min_value": self.min_value}


def _get_path(d: Dict[str, Any], path: str) -> Any:
    """Traverse dotted path like 'test.wape'."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check_thresholds(
    metrics: Dict[str, Any],
    thresholds: List[MetricThreshold],
) -> List[str]:
    """Return all failures. Empty list means all passed."""
    failures: List[str] = []
    for th in thresholds:
        msg = th.check(metrics)
        if msg is not None:
            failures.append(msg)
    return failures