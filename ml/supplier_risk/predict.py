"""Prediction wrapper.

Loads a fitted + calibrated model from an artifact directory and returns
per-(supplier, product, date) risk probabilities with provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from core.logging import get_logger
from data_platform.features.feature_registry import FEATURE_VERSION
from ml.supplier_risk.calibration import load_calibrator
from ml.supplier_risk.model import SupplierModelConfig, SupplierRiskModel

log = get_logger(__name__)


@dataclass
class SupplierRiskPrediction:
    supplier_code: str
    product_sku: Optional[str]
    date: str
    risk_probability: float
    risk_label: str
    threshold: float
    model_version: str
    feature_version: str
    horizon_days: int
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "supplier_code": self.supplier_code,
            "product_sku": self.product_sku,
            "date": self.date,
            "risk_probability": self.risk_probability,
            "risk_label": self.risk_label,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "horizon_days": self.horizon_days,
            "extras": self.extras,
        }


class SupplierRiskPredictor:
    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        meta_path = self.checkpoint_dir / "training_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata: {meta_path}")
        self.meta = json.loads(meta_path.read_text())

        self.feature_columns: List[str] = self.meta["feature_columns"]
        self.model_cfg = SupplierModelConfig(**self.meta["model_config"])
        self.model = SupplierRiskModel(self.model_cfg)
        self.model.load(str(self.checkpoint_dir / "model.json"))

        cal_path = self.checkpoint_dir / "calibrator.json"
        self.calibrator = load_calibrator(cal_path) if cal_path.exists() else None

        thr_path = self.checkpoint_dir / "threshold.json"
        self.threshold = 0.5
        if thr_path.exists():
            self.threshold = float(json.loads(thr_path.read_text())["threshold"])

        self.horizon_days = int(self.meta["horizon_days"])
        self.feature_version = self.meta.get("feature_version", FEATURE_VERSION)
        self.model_version = self.meta.get("model_version", "supplier_xgb_unnamed")

    # -- Batch prediction over a DataFrame ----------------------------
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with columns: supplier_code, product_sku, date,
        risk_probability, risk_label, threshold, model_version, feature_version.
        """
        if df.empty:
            return pd.DataFrame(columns=[
                "supplier_code", "product_sku", "date",
                "risk_probability", "risk_label",
                "threshold", "model_version", "feature_version",
            ])

        X = df[self.feature_columns].copy()
        # Mirror the train-time encoding: categoricals to string then numeric.
        for c in X.columns:
            if X[c].dtype == object:
                X[c] = X[c].fillna("__missing__").astype("category").cat.codes.astype("int32")
        for c in X.columns:
            if X[c].isna().any():
                X[c] = X[c].fillna(0.0)

        p_raw = self.model.predict_proba(X.to_numpy(dtype=np.float32))
        if self.calibrator is not None:
            p = self.calibrator.transform(p_raw)
        else:
            p = p_raw

        out = pd.DataFrame({
            "supplier_code": df["supplier_code"].values,
            "product_sku": df["product_sku"].values if "product_sku" in df.columns else None,
            "date": pd.to_datetime(df["date"]).dt.date.astype(str).values,
            "risk_probability": p,
            "threshold": self.threshold,
        })
        out["risk_label"] = np.where(out["risk_probability"] >= self.threshold, "high", "low")
        out["model_version"] = self.model_version
        out["feature_version"] = self.feature_version
        return out

    # -- Single prediction --------------------------------------------
    def predict_single(self, feature_row: pd.Series) -> SupplierRiskPrediction:
        df = feature_row.to_frame().T
        result = self.predict_dataframe(df)
        r = result.iloc[0]
        return SupplierRiskPrediction(
            supplier_code=str(r["supplier_code"]),
            product_sku=str(r["product_sku"]) if pd.notna(r.get("product_sku")) else None,
            date=str(r["date"]),
            risk_probability=float(r["risk_probability"]),
            risk_label=str(r["risk_label"]),
            threshold=float(r["threshold"]),
            model_version=str(r["model_version"]),
            feature_version=str(r["feature_version"]),
            horizon_days=self.horizon_days,
        )