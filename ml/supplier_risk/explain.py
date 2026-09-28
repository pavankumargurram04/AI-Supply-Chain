"""SHAP explanation for supplier risk.

Two entry points:

- explain_global: compute TreeSHAP over a sample of the training set and
  persist per-feature mean |SHAP| to CSV + a summary bar plot.
- explain_prediction: compute per-row SHAP values for one or a batch of
  predictions and return them as a DataFrame.

The SHAP package is imported lazily so a missing install degrades gracefully.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.logging import get_logger

log = get_logger(__name__)


def _import_shap():
    try:
        import shap  # type: ignore
        return shap
    except Exception as e:  # pragma: no cover - env dependent
        log.warning("shap.unavailable", extra={"error": str(e)})
        return None


def explain_global(
    model,                # SupplierRiskModel
    X: pd.DataFrame,
    *,
    output_dir: str | Path,
    run_name: str,
    sample_size: int = 2000,
) -> Optional[Dict[str, float]]:
    """Compute mean |SHAP| per feature and persist CSV + plot.

    Returns None if SHAP is unavailable.
    """
    shap = _import_shap()
    if shap is None or model.model is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_sample = X.sample(min(sample_size, len(X)), random_state=0)
    explainer = shap.TreeExplainer(model.model)
    values = explainer.shap_values(X_sample.to_numpy(dtype=np.float32))
    # Binary classifier: values is shape (N, F) for XGBoost >= 1.7.
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    mean_abs = np.mean(np.abs(values), axis=0)
    importance = {name: float(v) for name, v in zip(X.columns, mean_abs)}

    with (output_dir / f"shap_global_{run_name}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "mean_abs_shap"])
        for name, v in sorted(importance.items(), key=lambda kv: -kv[1]):
            w.writerow([name, v])

    # Bar plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sorted_items = sorted(importance.items(), key=lambda kv: kv[1])[-25:]
        names = [k for k, _ in sorted_items]
        vals = [v for _, v in sorted_items]
        fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(names))))
        ax.barh(names, vals)
        ax.set_xlabel("mean |SHAP|")
        fig.tight_layout()
        plots_dir = Path("reports/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots_dir / f"supplier_{run_name}_shap_global.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        log.warning("shap.plot_failed", extra={"error": str(e)})

    return importance


def explain_prediction(
    model,
    X: pd.DataFrame,
    *,
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Return per-row SHAP values as a DataFrame (rows indexed like X)."""
    shap = _import_shap()
    if shap is None or model.model is None:
        return None

    if max_rows is not None:
        X = X.head(max_rows)

    explainer = shap.TreeExplainer(model.model)
    values = explainer.shap_values(X.to_numpy(dtype=np.float32))
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    out = pd.DataFrame(values, columns=X.columns, index=X.index)
    out.insert(0, "_expected_value", float(explainer.expected_value)
               if not isinstance(explainer.expected_value, (list, np.ndarray))
               else float(np.asarray(explainer.expected_value).flatten()[-1]))
    return out