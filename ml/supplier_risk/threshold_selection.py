"""Validation-only threshold selection with a cost matrix.

Given calibrated probabilities on the validation set, choose the threshold
that minimises expected cost:

    cost = FN * cost_false_negative + FP * cost_false_positive

This function never sees the test set — that is enforced by the caller in
`train.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def select_threshold(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    cost_false_negative: float = 10.0,
    cost_false_positive: float = 1.0,
    grid: int = 200,
) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred, dtype=float)

    thresholds = np.linspace(0.01, 0.99, grid)
    best = {
        "threshold": 0.5,
        "cost": float("inf"),
        "fn": 0, "fp": 0, "tp": 0, "tn": 0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
    }
    curve: List[Dict] = []

    for thr in thresholds:
        y_hat = (p_pred >= thr).astype(int)
        fn = int(np.sum((y_true == 1) & (y_hat == 0)))
        fp = int(np.sum((y_true == 0) & (y_hat == 1)))
        tp = int(np.sum((y_true == 1) & (y_hat == 1)))
        tn = int(np.sum((y_true == 0) & (y_hat == 0)))
        cost = fn * cost_false_negative + fp * cost_false_positive

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        row = {
            "threshold": float(thr),
            "cost": float(cost),
            "fn": fn, "fp": fp, "tp": tp, "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        curve.append(row)
        if cost < best["cost"]:
            best = row

    return {
        **best,
        "cost_false_negative": float(cost_false_negative),
        "cost_false_positive": float(cost_false_positive),
        "curve": curve,
    }


def save_threshold(result: Dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Keep JSON small: store only the winner + a coarse curve sample.
    payload = {
        k: v for k, v in result.items()
        if k != "curve"
    }
    curve = result.get("curve", [])
    if curve:
        stride = max(1, len(curve) // 50)
        payload["curve_sample"] = curve[::stride]
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")