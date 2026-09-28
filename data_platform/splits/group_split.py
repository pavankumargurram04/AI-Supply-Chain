"""Group-aware split (e.g., hold out entire products or suppliers).

Useful for cold-start evaluation: a product in the test set has no training
history, forcing the model to rely on category / global signal.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def group_split(
    df: pd.DataFrame,
    *,
    group_col: str,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, holdout) where holdout contains whole groups."""
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not in DataFrame.")

    rng = np.random.default_rng(seed)
    groups = df[group_col].dropna().unique()
    rng.shuffle(groups)

    n_hold = max(1, int(round(len(groups) * holdout_fraction)))
    hold_groups = set(groups[:n_hold])

    hold = df[df[group_col].isin(hold_groups)].copy()
    train = df[~df[group_col].isin(hold_groups)].copy()
    return train, hold