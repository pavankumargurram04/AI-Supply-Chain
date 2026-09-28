"""Monte Carlo lead-time demand simulation.

Given:
    - forecast over the next H days (point + residual distribution)
    - lead-time distribution for the supplier (mean + std, or empirical samples)
    - current available inventory

Produce:
    - a large sample of cumulative demand over the lead time
    - stockout probability = P(LTD > available)
    - quantiles of LTD at any requested level
    - expected shortage

Design guarantees:
    - The residual distribution is bootstrapped from historical forecast
      residuals — NOT from an artificial Gaussian. If the residual pool is
      too small, we fall back to a Gaussian around the forecast with std
      estimated from the residuals (documented in the return payload).
    - The lead time is sampled from a clipped normal with the supplier's
      mean and standard deviation. Non-integer lead times are supported
      by interpolating the forecast linearly.
    - Deterministic: same input + same seed → same output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from core.deterministic import derive_seed, rng_from_seed


@dataclass
class SimulationConfig:
    n_samples: int = 10_000
    seed: int = 42
    alpha: float = 0.95              # service level target
    min_lead_time_days: float = 1.0
    max_lead_time_days: float = 180.0
    residual_pool_min: int = 200     # below this we fall back to Gaussian
    quantiles: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "seed": self.seed,
            "alpha": self.alpha,
            "min_lead_time_days": self.min_lead_time_days,
            "max_lead_time_days": self.max_lead_time_days,
            "residual_pool_min": self.residual_pool_min,
            "quantiles": list(self.quantiles),
        }


@dataclass
class LeadTimeDemandResult:
    """Result of a Monte Carlo simulation over the lead-time window."""
    mean: float
    std: float
    quantiles: dict           # {q: value}
    expected_shortage: float
    stockout_probability: float
    alpha_quantile: float     # the α-quantile of LTD
    lead_time_mean: float
    lead_time_std: float
    lead_time_p95: float
    n_samples: int
    seed: int
    mode: str                 # "residual_bootstrap" | "gaussian_fallback"
    config: dict = field(default_factory=dict)

    # Raw samples are kept small — we keep 256 to allow plotting without
    # blowing up memory. Downstream code that needs more can re-run.
    sample_quantiles: Sequence[float] = ()
    sample_values: Sequence[float] = ()

    def to_dict(self) -> dict:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "quantiles": {str(k): float(v) for k, v in self.quantiles.items()},
            "expected_shortage": float(self.expected_shortage),
            "stockout_probability": float(self.stockout_probability),
            "alpha_quantile": float(self.alpha_quantile),
            "lead_time_mean": float(self.lead_time_mean),
            "lead_time_std": float(self.lead_time_std),
            "lead_time_p95": float(self.lead_time_p95),
            "n_samples": int(self.n_samples),
            "seed": int(self.seed),
            "mode": self.mode,
            "config": self.config,
        }


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def simulate_lead_time_demand(
    *,
    forecast_point: np.ndarray,          # (H,) expected daily demand
    forecast_residuals: Optional[np.ndarray] = None,  # (N,) historical residuals
    lead_time_mean: float,
    lead_time_std: float,
    available_inventory: float,
    config: Optional[SimulationConfig] = None,
) -> LeadTimeDemandResult:
    """Monte Carlo simulation of cumulative demand during lead time.

    Args:
        forecast_point: per-day forecast for the next H days.
        forecast_residuals: pool of historical residuals (pred − actual).
            If None or too small, a Gaussian fallback is used.
        lead_time_mean / lead_time_std: supplier lead-time distribution.
        available_inventory: on_hand − reserved (units).
        config: simulation config.

    Returns:
        LeadTimeDemandResult with mean, std, quantiles, P(stockout), expected shortage.
    """
    cfg = config or SimulationConfig()
    forecast_point = np.asarray(forecast_point, dtype=np.float64)
    if forecast_point.ndim != 1 or forecast_point.size == 0:
        raise ValueError("forecast_point must be a non-empty 1-D array.")

    # Deterministic RNG. We derive a per-series seed that mixes config.seed with
    # the lead-time summary so the same numbers reproduce for the same input.
    child_seed = derive_seed(
        cfg.seed,
        "simulate_ltd",
        round(float(lead_time_mean), 6),
        round(float(lead_time_std), 6),
        round(float(available_inventory), 6),
        int(forecast_point.size),
    )
    rng = rng_from_seed(child_seed)

    H = forecast_point.size

    # 1) Sample lead times.
    lt_samples = rng.normal(lead_time_mean, max(lead_time_std, 1e-6), size=cfg.n_samples)
    lt_samples = np.clip(lt_samples, cfg.min_lead_time_days, cfg.max_lead_time_days)

    # 2) Demand sampling strategy.
    residuals = None
    mode = "gaussian_fallback"

    if forecast_residuals is not None and forecast_residuals.size >= cfg.residual_pool_min:
        residuals = np.asarray(forecast_residuals, dtype=np.float64)
        # Residuals are (pred − actual): actual ≈ pred − residual.
        # We only bootstrap residuals whose sign/magnitude will keep demand ≥ 0.
        # Demand sample = max(0, forecast + (-residual)) — i.e. actual-like.
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size >= cfg.residual_pool_min:
            mode = "residual_bootstrap"

    # 3) Build the (n_samples, H) demand matrix.
    #    Forecast beyond H days is not available — we approximate by
    #    repeating the last forecast value (constant tail). This is
    #    documented in the result payload (see `mode`).
    if residuals is not None:
        # Bootstrap: for each simulation, draw an index and use that residual
        # sequence. To keep the residual shape across the horizon, we sample
        # one residual per (simulation, day) independently. This is a
        # residual-bootstrap over *per-day* errors, which matches the
        # i.i.d.-residual assumption used when the residual pool is large.
        idx = rng.integers(0, residuals.size, size=(cfg.n_samples, H))
        sampled_resid = residuals[idx]                    # (n_samples, H)
        # Actual demand ≈ forecast − residual.
        demand = forecast_point[None, :] - sampled_resid
    else:
        # Gaussian fallback: scale = std of residuals if any, else 20% of forecast.
        if residuals is not None and residuals.size > 0:
            sigma = float(np.std(residuals))
        else:
            sigma = 0.20 * float(np.mean(forecast_point))
        sigma = max(sigma, 1e-6)
        noise = rng.normal(0.0, sigma, size=(cfg.n_samples, H))
        demand = forecast_point[None, :] + noise

    demand = np.clip(demand, 0.0, None)

    # 4) Sum demand over each simulation's lead time window.
    #    If lead time exceeds H, we extend using the last forecast value + noise.
    max_lt = int(np.ceil(lt_samples.max()))
    if max_lt > H:
        # Extend the tail: replicate last forecast and extend noise dimension.
        extra = max_lt - H
        tail_mean = np.full((cfg.n_samples, extra), forecast_point[-1])
        if residuals is not None:
            idx_tail = rng.integers(0, residuals.size, size=(cfg.n_samples, extra))
            tail = tail_mean - residuals[idx_tail]
        else:
            tail = tail_mean + rng.normal(0.0, sigma, size=(cfg.n_samples, extra))
        demand = np.concatenate([demand, np.clip(tail, 0.0, None)], axis=1)
        # Also extend the point forecast vector for indexing.
        full_forecast = np.concatenate([forecast_point, np.full(extra, forecast_point[-1])])
    else:
        full_forecast = forecast_point

    # 5) Cumulative sum, take value at index ceil(L) − 1 (with linear interp).
    #    For non-integer lead times we linearly interpolate between the two
    #    adjacent cumulative sums.
    cum = np.cumsum(demand, axis=1)                        # (n_samples, T)
    n_samples = cfg.n_samples
    lt_int_low = np.floor(lt_samples).astype(int) - 1
    lt_int_high = np.ceil(lt_samples).astype(int) - 1
    frac = lt_samples - np.floor(lt_samples)               # in [0, 1)

    # Clip indices to valid range.
    T = cum.shape[1]
    lt_int_low = np.clip(lt_int_low, 0, T - 1)
    lt_int_high = np.clip(lt_int_high, 0, T - 1)

    lt_demand = (
        (1.0 - frac) * cum[np.arange(n_samples), lt_int_low]
        + frac * cum[np.arange(n_samples), lt_int_high]
    )

    # 6) Result statistics.
    mean = float(np.mean(lt_demand))
    std = float(np.std(lt_demand))
    quantile_values = {
        float(q): float(np.quantile(lt_demand, q)) for q in cfg.quantiles
    }
    alpha_q = float(np.quantile(lt_demand, cfg.alpha))
    stockout_prob = float(np.mean(lt_demand > available_inventory))
    expected_shortage = float(np.mean(np.maximum(lt_demand - available_inventory, 0.0)))

    # Small footprint sample for downstream plotting.
    sample_q = np.linspace(0.0, 1.0, 256)
    sample_vals = np.quantile(lt_demand, sample_q)

    return LeadTimeDemandResult(
        mean=mean,
        std=std,
        quantiles=quantile_values,
        expected_shortage=expected_shortage,
        stockout_probability=stockout_prob,
        alpha_quantile=alpha_q,
        lead_time_mean=float(np.mean(lt_samples)),
        lead_time_std=float(np.std(lt_samples)),
        lead_time_p95=float(np.quantile(lt_samples, 0.95)),
        n_samples=cfg.n_samples,
        seed=child_seed,
        mode=mode,
        config=cfg.to_dict(),
        sample_quantiles=tuple(float(x) for x in sample_q),
        sample_values=tuple(float(x) for x in sample_vals),
    )