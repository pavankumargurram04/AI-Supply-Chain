"""Demand forecast endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from apps.api.dependencies import get_current_user, get_db, require_analyst
from apps.api.schemas import ForecastOut
from apps.api.services import curated_features, demand_checkpoint
from core.errors import ModelError

router = APIRouter(prefix="/forecast", tags=["forecast"])


class PredictIn(BaseModel):
    product_sku: str
    warehouse_code: str
    mode: str = "dev"
    origin_date: Optional[str] = None
    mc_dropout_samples: int = Field(default=0, ge=0, le=500)


@router.get("/{product_sku}/{warehouse_code}", response_model=ForecastOut,
            dependencies=[Depends(get_current_user)])
def get_forecast(
    product_sku: str,
    warehouse_code: str,
    mode: str = Query(default="dev"),
    db: Session = Depends(get_db),
) -> ForecastOut:
    """Forecast using the last available complete window."""
    import pandas as pd
    from ml.demand.predict import DemandPredictor

    ckpt = demand_checkpoint(mode)
    feature_file = next(curated_features(mode).glob("demand_features_*.parquet"), None)
    if feature_file is None:
        raise ModelError(f"No demand feature file under {curated_features(mode)}")
    df = pd.read_parquet(feature_file)

    predictor = DemandPredictor(ckpt, device="cpu")
    try:
        fc = predictor.predict(
            df,
            product_sku=product_sku,
            warehouse_code=warehouse_code,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ForecastOut(**fc.to_dict())


@router.post("/predict", response_model=ForecastOut,
             dependencies=[Depends(require_analyst)])
def predict(payload: PredictIn) -> ForecastOut:
    """Explicit predict with optional origin date and MC-dropout sampling."""
    import pandas as pd
    from ml.demand.predict import DemandPredictor

    ckpt = demand_checkpoint(payload.mode)
    feature_file = next(curated_features(payload.mode).glob("demand_features_*.parquet"), None)
    if feature_file is None:
        raise ModelError("No demand feature file")
    df = pd.read_parquet(feature_file)

    predictor = DemandPredictor(ckpt, device="cpu")
    try:
        fc = predictor.predict(
            df,
            product_sku=payload.product_sku,
            warehouse_code=payload.warehouse_code,
            origin_date=payload.origin_date,
            mc_dropout_samples=payload.mc_dropout_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ForecastOut(**fc.to_dict())
