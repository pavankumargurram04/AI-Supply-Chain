"""Deterministic execution helpers.

Every random process (data generation, training, Monte Carlo simulation)
must call `seed_everything()` with an explicit, recorded seed.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
from typing import Iterator, Optional

import numpy as np


def _seed_python(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _seed_numpy(seed: int) -> None:
    np.random.seed(seed)


def _seed_torch(seed: int) -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may slow training slightly).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed every relevant random generator. Returns the seed used."""
    _seed_python(seed)
    _seed_numpy(seed)
    _seed_torch(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            import torch  # type: ignore
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    return seed


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable child seed from a base seed and a set of parts.

    Used so that independent subsystems (demand, supplier, inventory) each
    receive a deterministic but different seed from one master seed.
    """
    payload = "|".join([str(base_seed), *[str(p) for p in parts]]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


@contextlib.contextmanager
def temporary_seed(seed: int) -> Iterator[int]:
    """Context manager that seeds and restores Python/NumPy RNG state."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        seed_everything(seed, deterministic=False)
        yield seed
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def rng_from_seed(seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy Generator. Prefer this over legacy np.random."""
    if seed is None:
        from core.config import get_settings
        seed = get_settings().RANDOM_SEED
    return np.random.default_rng(seed)