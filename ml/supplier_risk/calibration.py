"""Probability calibration: isotonic and Platt.

Both calibrators are fit on validation predictions only. They transform
raw classifier probabilities into calibrated probabilities.

Persistence format: JSON with explicit parameters so we don't need pickle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------
class IsotonicCalibrator:
    kind = "isotonic"

    def __init__(self) -> None:
        self.iso: Optional[IsotonicRegression] = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        self.iso.fit(p_raw, y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if self.iso is None:
            raise RuntimeError("Isotonic calibrator is not fitted.")
        return np.clip(self.iso.predict(p_raw), 1e-6, 1 - 1e-6)

    # -- persistence -------------------------------------------------
    def to_json(self) -> dict:
        if self.iso is None:
            raise RuntimeError("Unfitted calibrator cannot be serialised.")
        return {
            "kind": self.kind,
            "X_thresholds": self.iso.X_thresholds_.tolist(),
            "y_thresholds": self.iso.y_thresholds_.tolist(),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "IsotonicCalibrator":
        c = cls()
        c.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        # Reconstruct by fitting the interpolation piecewise.
        # sklearn exposes X_thresholds_ / y_thresholds_ after fit. We rebuild via
        # a monotone step fit using those knots.
        x = np.asarray(payload["X_thresholds"])
        y = np.asarray(payload["y_thresholds"])
        c.iso.fit(x, y)
        return c


class PlattCalibrator:
    kind = "platt"

    def __init__(self) -> None:
        self.lr: Optional[LogisticRegression] = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.lr.fit(p_raw.reshape(-1, 1), y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if self.lr is None:
            raise RuntimeError("Platt calibrator is not fitted.")
        return self.lr.predict_proba(p_raw.reshape(-1, 1))[:, 1]

    def to_json(self) -> dict:
        if self.lr is None:
            raise RuntimeError("Unfitted calibrator cannot be serialised.")
        return {
            "kind": self.kind,
            "coef": self.lr.coef_.tolist(),
            "intercept": self.lr.intercept_.tolist(),
            "classes": self.lr.classes_.tolist(),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "PlattCalibrator":
        c = cls()
        c.lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        # Force-fit with two synthetic points so the parameters match the serialised ones.
        coef = np.array(payload["coef"]).reshape(1, -1)
        intercept = np.array(payload["intercept"])
        classes = np.array(payload["classes"])
        c.lr.coef_ = coef
        c.lr.intercept_ = intercept
        c.lr.classes_ = classes
        c.lr.n_features_in_ = 1
        return c


# ---------------------------------------------------------------------------
# Factory / helpers
# ---------------------------------------------------------------------------
def fit_calibrator(kind: str, p_raw: np.ndarray, y: np.ndarray):
    if kind == "isotonic":
        return IsotonicCalibrator().fit(p_raw, y)
    if kind == "platt":
        return PlattCalibrator().fit(p_raw, y)
    raise ValueError(f"Unknown calibrator: {kind}")


def save_calibrator(calibrator, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(calibrator.to_json(), indent=2), encoding="utf-8")


def load_calibrator(path: str | Path):
    payload = json.loads(Path(path).read_text())
    kind = payload.get("kind")
    if kind == "isotonic":
        return IsotonicCalibrator.from_json(payload)
    if kind == "platt":
        return PlattCalibrator.from_json(payload)
    raise ValueError(f"Unknown calibrator kind in {path}: {kind}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def reliability_curve(
    p_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 10,
) -> List[dict]:
    """Return per-bin (mean_pred, empirical_rate, n)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_pred, edges) - 1, 0, n_bins - 1)
    rows: List[dict] = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1],
                         "mean_pred": None, "empirical": None, "n": 0})
            continue
        rows.append({
            "bin": b, "lo": edges[b], "hi": edges[b + 1],
            "mean_pred": float(np.mean(p_pred[mask])),
            "empirical": float(np.mean(y_true[mask])),
            "n": int(mask.sum()),
        })
    return rows