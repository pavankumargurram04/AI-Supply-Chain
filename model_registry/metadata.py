"""Model cards.

A model card is a JSON document describing a model version in human-readable
form. It is stored alongside the model artifact in the object store, and the
metadata is also written to the `model_registry.extras` JSON column.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelCard:
    model_name: str
    model_family: str
    version: str
    dataset_version: Optional[str]
    feature_version: Optional[str]
    trained_at: str
    trained_by: str
    framework: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    intended_use: str = ""
    limitations: List[str] = field(default_factory=list)
    training_data_summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return p


def build_model_card(
    *,
    model_name: str,
    model_family: str,
    version: str,
    dataset_version: Optional[str],
    feature_version: Optional[str],
    framework: str,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, Any],
    trained_by: str,
    training_data_summary: Optional[Dict[str, Any]] = None,
    intended_use: str = "",
    limitations: Optional[List[str]] = None,
    notes: Optional[List[str]] = None,
) -> ModelCard:
    """Build a ModelCard with sensible defaults for each family."""
    default_intended = {
        "demand_lstm": "Global multi-series daily demand forecasting.",
        "supplier_xgb": "Predict supplier disruption risk within a configured horizon.",
        "inventory_dqn": "Inventory replenishment policy; outputs a multiplier of expected lead-time demand.",
        "procurement_opt": "Procurement allocation and quantity optimization.",
    }
    default_limitations = {
        "demand_lstm": [
            "Trained on the provided historical window; performance degrades outside it.",
            "Requires at least lookback+horizon contiguous rows per series.",
            "Synthetic-benchmark performance must not be assumed to transfer to any specific organization.",
        ],
        "supplier_xgb": [
            "Risk labels are defined by a future-event window; changing the horizon changes the task.",
            "Calibration fit on the validation split; re-evaluate after any retraining.",
        ],
        "inventory_dqn": [
            "Policy is trained against a specific cost model; transfer to other cost regimes requires retraining.",
            "Action space is discrete; continuous review is only approximated.",
        ],
        "procurement_opt": [
            "Greedy risk-adjusted allocation; a MILP solver may improve cost under joint constraints.",
        ],
    }

    return ModelCard(
        model_name=model_name,
        model_family=model_family,
        version=version,
        dataset_version=dataset_version,
        feature_version=feature_version,
        trained_at=datetime.now(tz=timezone.utc).isoformat(),
        trained_by=trained_by,
        framework=framework,
        hyperparameters=hyperparameters,
        metrics=metrics,
        intended_use=intended_use or default_intended.get(model_family, ""),
        limitations=limitations if limitations is not None else default_limitations.get(model_family, []),
        training_data_summary=training_data_summary or {},
        notes=notes or [],
    )