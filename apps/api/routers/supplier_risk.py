"""Supplier risk endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db, require_analyst
from apps.api.schemas import SupplierRiskOut
from apps.api.services import curated_features, supplier_checkpoint
from core.errors import ModelError

router = APIRouter(prefix="/supplier-risk", tags=["supplier-risk"])


class PredictSupplierRiskIn(BaseModel):
    product_sku: str
    mode: str = "dev"


@router.get("", response_model=list[SupplierRiskOut],
            dependencies=[Depends(get_current_user)])
def risk_for_product(
    product_sku: str = Query(...),
    mode: str = Query(default="dev"),
    db: Session = Depends(get_db),
) -> list[SupplierRiskOut]:
    """Score every supplier candidate for a product using the latest features."""
    import pandas as pd
    from ml.supplier_risk.predict import SupplierRiskPredictor

    ckpt = supplier_checkpoint(mode)
    feat_file = next(curated_features(mode).glob("supplier_features_*.parquet"), None)
    if feat_file is None:
        raise ModelError("No supplier feature file")
    df = pd.read_parquet(feat_file)
    sub = df[df["product_sku"] == product_sku].copy()
    if sub.empty:
        return []
    latest = sub.sort_values("date").groupby("supplier_code", as_index=False).tail(1)
    predictor = SupplierRiskPredictor(ckpt)
    out = predictor.predict_dataframe(latest)
    return [
        SupplierRiskOut(
            supplier_code=str(r["supplier_code"]),
            risk_probability=float(r["risk_probability"]),
            risk_label=str(r["risk_label"]),
            lead_time_days=float(latest.iloc[i].get("feat_sup_lead_time_days", 0.0)),
            unit_price=float(latest.iloc[i].get("feat_sup_unit_price", 0.0)),
            moq=int(latest.iloc[i].get("feat_sup_moq", 1)),
            is_preferred=False,
            model_version=str(r["model_version"]),
        )
        for i, (_, r) in enumerate(out.iterrows())
    ]


@router.post("/predict", response_model=list[SupplierRiskOut],
             dependencies=[Depends(require_analyst)])
def predict(payload: PredictSupplierRiskIn) -> list[SupplierRiskOut]:
    return risk_for_product(product_sku=payload.product_sku, mode=payload.mode)