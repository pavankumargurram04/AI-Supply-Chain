"""Supplier risk classifier (XGBoost + SHAP + calibration).

Public API:
    from ml.supplier_risk.dataset import build_supplier_dataset, SupplierDatasetBundle
    from ml.supplier_risk.train   import train_supplier_model, SupplierTrainConfig
    from ml.supplier_risk.predict import SupplierRiskPredictor
    from ml.supplier_risk.explain import explain_prediction, explain_global
"""

from ml.supplier_risk.dataset import (
    SupplierDatasetBundle,
    build_supplier_dataset,
    DISRUPTIVE_EVENT_TYPES,
)
from ml.supplier_risk.model import SupplierModelConfig, SupplierRiskModel
from ml.supplier_risk.train import SupplierTrainConfig, train_supplier_model

__all__ = [
    "SupplierDatasetBundle",
    "build_supplier_dataset",
    "DISRUPTIVE_EVENT_TYPES",
    "SupplierModelConfig",
    "SupplierRiskModel",
    "SupplierTrainConfig",
    "train_supplier_model",
]