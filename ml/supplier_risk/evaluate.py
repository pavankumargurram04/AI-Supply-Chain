"""Binary classification metrics for supplier risk.

All metrics are computed against *calibrated* probabilities where possible.
ROC-AUC and PR-AUC are threshold-free. F1 / precision / recall are evaluated
at a caller-provided threshold. Brier and log-loss are proper scoring rules
that benefit from calibration.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.clip(np.asarray(p_pred, dtype=float), 1e-9, 1.0 - 1e-9)

    out: Dict[str, float] = {}
    out["n"] = int(len(y_true))
    out["positive_rate"] = float(np.mean(y_true)) if len(y_true) else 0.0
    out["mean_pred"] = float(np.mean(p_pred)) if len(p_pred) else 0.0

    if len(np.unique(y_true)) < 2:
        out.update({
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": float(np.mean((p_pred - y_true) ** 2)),
            "log_loss": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        })
        return out

    y_pred = (p_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out.update({
        "roc_auc": float(roc_auc_score(y_true, p_pred)),
        "pr_auc": float(average_precision_score(y_true, p_pred)),
        "brier": float(brier_score_loss(y_true, p_pred)),
        "log_loss": float(log_loss(y_true, p_pred, labels=[0, 1])),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(threshold),
    })
    return out