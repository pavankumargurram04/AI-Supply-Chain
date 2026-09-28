"""Service layer bridging routers and the decision engine / agents.

Routers must stay thin. Every model/agent call lives here. Each service
resolves the artifacts for the active mode and raises a typed `ModelError`
if a needed checkpoint is missing, so the API returns 500 with a clear
message rather than a stack trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from core.errors import ModelError, ValidationError


_DEFAULT_MODE = "dev"


def _resolve_mode(mode: Optional[str]) -> str:
    m = mode or _DEFAULT_MODE
    if m not in ("dev", "research", "stress"):
        raise ValidationError(f"Unknown mode: {m}")
    return m


def _checkpoint_path(kind: str, mode: str) -> Path:
    p = Path("artifacts") / kind / mode
    return p


def demand_checkpoint(mode: Optional[str] = None) -> Path:
    m = _resolve_mode(mode)
    p = _checkpoint_path("demand", m)
    if not (p / "training_metadata.json").exists():
        raise ModelError(
            f"Demand checkpoint not found at {p}. "
            f"Run: python scripts\\train_all.py --mode {m} --model demand"
        )
    return p


def supplier_checkpoint(mode: Optional[str] = None) -> Path:
    m = _resolve_mode(mode)
    p = _checkpoint_path("supplier", m)
    if not (p / "training_metadata.json").exists():
        raise ModelError(
            f"Supplier checkpoint not found at {p}. "
            f"Run: python scripts\\train_all.py --mode {m} --model supplier"
        )
    return p


def inventory_checkpoint(mode: Optional[str] = None) -> Optional[Path]:
    m = _resolve_mode(mode)
    p = _checkpoint_path("inventory", m)
    return p if (p / "training_metadata.json").exists() else None


def curated_features(mode: Optional[str] = None) -> Path:
    m = _resolve_mode(mode)
    p = Path("datasets") / "curated" / m / "features"
    if not p.exists():
        raise ModelError(
            f"Curated features not found at {p}. "
            f"Run: python scripts\\build_features.py --mode {m}"
        )
    return p


def raw_dataset(mode: Optional[str] = None) -> Path:
    m = _resolve_mode(mode)
    p = Path("datasets") / "raw" / m
    if not p.exists():
        raise ModelError(
            f"Raw dataset not found at {p}. "
            f"Run: python scripts\\generate_synthetic_data.py --mode {m}"
        )
    return p


def build_decision_engine(mode: Optional[str] = None, *, device: str = "cpu",
                          simulation_samples: int = 10_000):
    """Construct a DecisionEngine for the requested mode."""
    from decision_engine.engine import DecisionEngine

    m = _resolve_mode(mode)
    return DecisionEngine(
        demand_checkpoint_dir=demand_checkpoint(m),
        supplier_checkpoint_dir=supplier_checkpoint(m),
        inventory_checkpoint_dir=inventory_checkpoint(m),
        curated_features_dir=curated_features(m),
        raw_dir=raw_dataset(m),
        simulation_samples=simulation_samples,
        device=device,
    )


def build_orchestrator(mode: Optional[str] = None, *, device: str = "cpu",
                       simulation_samples: int = 10_000):
    """Construct an AgentOrchestrator for the requested mode."""
    from agents.coordinator_agent import AgentOrchestrator

    m = _resolve_mode(mode)
    return AgentOrchestrator(
        demand_checkpoint_dir=demand_checkpoint(m),
        supplier_checkpoint_dir=supplier_checkpoint(m),
        inventory_checkpoint_dir=inventory_checkpoint(m),
        curated_features_dir=curated_features(m),
        raw_dir=raw_dataset(m),
        simulation_samples=simulation_samples,
        device=device,
    )