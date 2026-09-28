"""Uncertainty helpers.

Two supported strategies, both deterministic given the seed:

1. Quantile head (preferred). If the model was trained with
   `quantiles=(0.1, 0.5, 0.9)` (or similar), we just read off the intervals.
   No stochastic sampling required.

2. MC-dropout. Turn dropout on, sample S predictions, take percentiles.
   Deterministic if `torch.manual_seed(seed)` is set before sampling.

The public functions here operate on a `DemandPredictor` and return numpy
arrays. They are used by `predict.py` and by the decision engine's downstream
stockout simulation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from core.deterministic import seed_everything
from ml.demand.predict import DemandPredictor


def mc_dropout_interval(
    predictor: DemandPredictor,
    *,
    enc_t: torch.Tensor,
    dec_t: torch.Tensor,
    static_cat_t: torch.Tensor,
    static_num_t: torch.Tensor,
    samples: int = 100,
    lower_q: float = 0.1,
    upper_q: float = 0.9,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (point_median, lower, upper) all in real units, shape (H,)."""
    seed_everything(seed, deterministic=False)
    predictor.model.train()
    preds = []
    try:
        with torch.no_grad():
            for _ in range(samples):
                pred = predictor.model(enc_t, dec_t, static_cat_t, static_num_t)
                if predictor.quantiles is not None:
                    q_list = list(predictor.quantiles)
                    med_idx = q_list.index(0.5) if 0.5 in q_list else len(q_list) // 2
                    pred = pred[:, :, med_idx]
                preds.append(predictor.scalers.inverse_target(pred.detach().cpu().numpy())[0])
    finally:
        predictor.model.eval()
    stacked = np.stack(preds, axis=0)   # (S, H)
    point = np.median(stacked, axis=0)
    lower = np.percentile(stacked, lower_q * 100.0, axis=0)
    upper = np.percentile(stacked, upper_q * 100.0, axis=0)
    return point, lower, upper


def interval_width(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return upper - lower