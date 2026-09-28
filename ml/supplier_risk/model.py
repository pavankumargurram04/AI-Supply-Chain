"""XGBoost classifier wrapper with reproducible training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import xgboost as xgb


@dataclass
class SupplierModelConfig:
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    tree_method: str = "hist"
    objective: str = "binary:logistic"
    eval_metric: str = "aucpr"
    n_jobs: int = 0
    early_stopping_rounds: Optional[int] = 30

    def to_dict(self) -> dict:
        return asdict(self)


class SupplierRiskModel:
    """Thin wrapper so we can save/load with our own metadata."""

    def __init__(self, cfg: SupplierModelConfig) -> None:
        self.cfg = cfg
        params = cfg.to_dict().copy()
        params.pop("n_estimators", None)
        params.pop("early_stopping_rounds", None)
        # XGBoost handles these at fit / predict time.
        self._booster_params = params
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_iteration: Optional[int] = None
        self.evals_result: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SupplierRiskModel":
        cfg = self.cfg
        self.model = xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            gamma=cfg.gamma,
            tree_method=cfg.tree_method,
            objective=cfg.objective,
            eval_metric=cfg.eval_metric,
            n_jobs=cfg.n_jobs,
            early_stopping_rounds=cfg.early_stopping_rounds,
            random_state=0,   # pinned externally by seed_everything
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
            sample_weight=sample_weight,
        )
        self.best_iteration = getattr(self.model, "best_iteration", None)
        self.evals_result = getattr(self.model, "evals_result", lambda: {})()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        # Use best_iteration if early stopping was used.
        kwargs = {}
        if self.best_iteration is not None:
            kwargs["iteration_range"] = (0, self.best_iteration + 1)
        return self.model.predict_proba(X, **kwargs)[:, 1]

    def feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        booster = self.model.get_booster()
        scores = booster.get_score(importance_type="gain")
        # Map to names.
        out = {name: float(scores.get(f"f{i}", 0.0))
               for i, name in enumerate(feature_names)}
        return out

    # -- persistence --------------------------------------------------
    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save unfitted model.")
        self.model.save_model(path)

    def load(self, path: str) -> "SupplierRiskModel":
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        self.best_iteration = getattr(self.model, "best_iteration", None)
        return self