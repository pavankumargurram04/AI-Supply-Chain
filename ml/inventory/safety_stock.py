"""Safety stock derivations.

Two independent approaches; both are documented and used in ablation.

1. `safety_stock_from_simulation`: uses the α-quantile of the Monte Carlo
   lead-time-demand distribution. This is the *primary* method used by the
   decision engine because it makes no normality assumptions.

2. `safety_stock_normal_approx`: classic formula
       SS = z(α) * sqrt(L * σ_D^2 + D̄^2 * σ_L^2)
   provided as a *baseline* for the ablation study. It is never used as a
   hidden decision rule.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ml.inventory.simulation import LeadTimeDemandResult


def _z_from_alpha(alpha: float) -> float:
    """Standard-normal quantile (inverse CDF).

    Implemented via a rational approximation (Acklam). Adequate for 0.5 < α < 0.9999.
    """
    if not 0.5 < alpha < 0.9999:
        raise ValueError("alpha must be in (0.5, 0.9999)")
    # Coefficients from Acklam's algorithm.
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if alpha < plow:
        q = math.sqrt(-2.0 * math.log(alpha))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if alpha > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - alpha))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = alpha - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


def safety_stock_from_simulation(
    ltd_result: LeadTimeDemandResult,
    *,
    service_level: float,
) -> float:
    """Safety stock = Q_α(LTD) − E[LTD].

    Because `ltd_result` already contains quantiles computed from the empirical
    distribution, we simply look up (or interpolate) the target quantile and
    subtract the mean.
    """
    if service_level in ltd_result.quantiles:
        q_alpha = ltd_result.quantiles[service_level]
    else:
        # Interpolate from the stored sample quantiles.
        qs = np.asarray(ltd_result.sample_quantiles, dtype=float)
        vs = np.asarray(ltd_result.sample_values, dtype=float)
        q_alpha = float(np.interp(service_level, qs, vs))
    ss = q_alpha - ltd_result.mean
    return max(0.0, float(ss))


def safety_stock_normal_approx(
    *,
    mean_daily_demand: float,
    std_daily_demand: float,
    lead_time_mean: float,
    lead_time_std: float,
    service_level: float,
) -> float:
    """Classic formula. Returned value is the *safety stock above expected
    lead-time demand* — i.e., the buffer over the mean.
    """
    z = _z_from_alpha(service_level)
    variance = lead_time_mean * (std_daily_demand ** 2) + (mean_daily_demand ** 2) * (lead_time_std ** 2)
    return max(0.0, float(z * math.sqrt(max(variance, 0.0))))