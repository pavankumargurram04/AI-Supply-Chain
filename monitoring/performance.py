"""Time-resolved performance tracking.

Given a series of (timestamp, value) and (timestamp, reference) rows, we
bucket into windows and report a rolling metric. This is used by the demand
and supplier monitors to detect performance regressions over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class PerformanceWindow:
    start: str
    end: str
    n: int
    metrics: Dict[str, float]

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "n": self.n, "metrics": self.metrics}


def _mae(p, t):   return float(np.mean(np.abs(p - t)))
def _rmse(p, t):  return float(np.sqrt(np.mean((p - t) ** 2)))
def _wape(p, t):  return float(np.sum(np.abs(p - t)) / (np.sum(np.abs(t)) + 1e-9))
def _bias(p, t):  return float(np.mean(p - t))


def rolling_performance(
    df: pd.DataFrame,
    *,
    date_col: str,
    pred_col: str,
    target_col: str,
    freq: str = "30D",
    metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
) -> List[PerformanceWindow]:
    """Bucket rows into calendar windows and compute metrics per window.

    freq uses pandas offset aliases ("7D", "30D", "1M").
    """
    if df.empty:
        return []
    metrics = metrics or {
        "mae": _mae, "rmse": _rmse, "wape": _wape, "bias": _bias,
    }

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    out: List[PerformanceWindow] = []
    for start, group in d.groupby(pd.Grouper(key=date_col, freq=freq)):
        if group.empty:
            continue
        p = group[pred_col].to_numpy(dtype=float)
        t = group[target_col].to_numpy(dtype=float)
        vals = {name: float(fn(p, t)) for name, fn in metrics.items()}
        out.append(PerformanceWindow(
            start=start.date().isoformat(),
            end=(start + pd.tseries.frequencies.to_offset(freq) - pd.Timedelta(days=1)).date().isoformat(),
            n=int(len(group)),
            metrics=vals,
        ))
    return out


def trend(
    windows: Sequence[PerformanceWindow],
    *,
    metric: str,
) -> Optional[Dict[str, float]]:
    """Return a simple linear slope + delta between first and last window."""
    if len(windows) < 2:
        return None
    vals = [w.metrics.get(metric) for w in windows]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    x = np.arange(len(vals), dtype=float)
    y = np.asarray(vals, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "slope": float(slope),
        "first": float(vals[0]),
        "last": float(vals[-1]),
        "delta": float(vals[-1] - vals[0]),
        "mean": float(np.mean(vals)),
    }