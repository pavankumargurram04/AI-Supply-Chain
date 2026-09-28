"""Supplier risk decision step.

Given the DecisionState candidates, call the supplier risk predictor with
the most recent feature row for each candidate. Returns the candidates with
risk filled in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from core.logging import get_logger
from decision_engine.state_builder import DecisionState, SupplierCandidate

log = get_logger(__name__)


def score_suppliers(
    state: DecisionState,
    *,
    supplier_checkpoint_dir: str | Path,
) -> List[Dict[str, Any]]:
    if not state.suppliers:
        return []

    ckpt = Path(supplier_checkpoint_dir)
    if not ckpt.exists():
        log.warning("decision.supplier.no_checkpoint", extra={"path": str(ckpt)})
        # Fall through with default risk of 0.5 for every candidate.
        return [
            {"supplier_code": s.supplier_code, "risk_probability": 0.5,
             "risk_label": "unknown", "model_version": "none"}
            for s in state.suppliers
        ]

    from ml.supplier_risk.predict import SupplierRiskPredictor

    predictor = SupplierRiskPredictor(ckpt)

    # Use the latest feature row per supplier.
    df = state.supplier_feature_frame
    if df.empty:
        log.warning("decision.supplier.no_features")
        return [
            {"supplier_code": s.supplier_code, "risk_probability": 0.5,
             "risk_label": "unknown", "model_version": predictor.model_version}
            for s in state.suppliers
        ]

    latest = (
        df.sort_values("date")
        .groupby("supplier_code", as_index=False)
        .tail(1)
    )
    preds = predictor.predict_dataframe(latest)
    risk_map = dict(zip(preds["supplier_code"], preds["risk_probability"]))
    label_map = dict(zip(preds["supplier_code"], preds["risk_label"]))

    out: List[Dict[str, Any]] = []
    for s in state.suppliers:
        risk = float(risk_map.get(s.supplier_code, 0.5))
        s.risk_probability = risk
        s.risk_label = label_map.get(s.supplier_code, "unknown")
        out.append({
            "supplier_code": s.supplier_code,
            "risk_probability": risk,
            "risk_label": s.risk_label,
            "reliability_class": s.reliability_class,
            "lead_time_days": float(s.lead_time_days),
            "lead_time_std_days": float(s.lead_time_std_days),
            "unit_price": float(s.unit_price),
            "moq": int(s.moq),
            "max_order_qty": int(s.max_order_qty),
            "shipping_cost": float(s.shipping_cost),
            "is_preferred": bool(s.is_preferred),
            "model_version": predictor.model_version,
        })
    return out