"""Celery forecasting tasks (batch)."""

from __future__ import annotations

from typing import Any, Dict, List

from core.logging import get_logger
from apps.worker.celery_app import celery_app

log = get_logger(__name__)


@celery_app.task(name="tasks.forecast.batch")
def batch_forecast(product_skus: List[str], warehouse_codes: List[str],
                   mode: str = "dev") -> Dict[str, Any]:
    """Batch-forecast every (product, warehouse) combination and return point estimates."""
    import pandas as pd
    from ml.demand.predict import DemandPredictor

    feature_file = next(
        (__import__("pathlib").Path(f"datasets/curated/{mode}/features")).glob("demand_features_*.parquet"),
        None,
    )
    if feature_file is None:
        return {"ok": False, "error": "features not found"}
    df = pd.read_parquet(feature_file)
    predictor = DemandPredictor(f"artifacts/demand/{mode}")

    results: Dict[str, Any] = {"ok": True, "forecasts": {}}
    for sku in product_skus:
        for wh in warehouse_codes:
            try:
                fc = predictor.predict(df, product_sku=sku, warehouse_code=wh)
                results["forecasts"][f"{sku}|{wh}"] = {"point": fc.point,
                                                       "model_version": fc.model_version}
            except Exception as e:  # noqa: BLE001
                results["forecasts"][f"{sku}|{wh}"] = {"error": str(e)}
    return results