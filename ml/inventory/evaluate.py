"""Head-to-head comparison: DQN vs baselines.

Produces reports/metrics/inventory_eval_<run>.json with per-baseline and DQN
metrics, and reports/plots/inventory_<run>_comparison.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.logging import configure_logging, get_logger
from ml.inventory.baselines import ALL_BASELINES, run_baseline
from ml.inventory.cost_model import CostParameters
from ml.inventory.policy import InventoryPolicy
from ml.inventory.train import _load_series

log = get_logger(__name__)


def _run_dqn(
    series: dict,
    *,
    checkpoint_dir: Path,
    cost: CostParameters,
    episode_days: int = 60,
    start_index: int = 0,
) -> dict:
    policy = InventoryPolicy(checkpoint_dir, device="cpu")
    from ml.inventory.environment import EnvConfig, InventoryEnvironment

    env_cfg = EnvConfig(cost=cost, episode_days=episode_days)
    env = InventoryEnvironment(env_cfg)
    env.attach_series(
        forecast_point=series["forecast_point"],
        forecast_std=series["forecast_std"],
        actual_demand=series["actual_demand"],
        lead_time_mean=series["lead_time_mean"],
        lead_time_std=series["lead_time_std"],
        supplier_risk=series["supplier_risk"],
        initial_inventory=float(np.mean(series["forecast_point"][:30]) * 30.0),
        seed=42,
    )
    state = env.reset(start_index=start_index)
    done = False
    while not done:
        expected_ltd = float(env.forecast_point[env.t] * env.lead_time_mean[env.t])
        decision = policy.decide_from_state(state, expected_lead_time_demand=expected_ltd)
        state, _r, done, _ = env.step(decision.action_index)
    return env.episode_summary()


def run_inventory_eval(
    *,
    features_dir: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    run_name: str,
    max_series: int = 5,
    episode_days: int = 60,
    seed: int = 42,
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_list = _load_series(
        Path(features_dir),
        supplier_artifacts_dir=None,
        max_series=max_series,
        seed=seed,
    )
    if not series_list:
        raise ValueError("No series to evaluate.")

    unit_cost = float(series_list[0]["unit_cost"])
    cost = CostParameters(
        unit_cost=unit_cost,
        holding_rate_annual=0.20,
        order_fixed_cost=250.0,
        shortage_cost_per_unit=unit_cost * 3.0,
        shipping_cost_per_order=50.0,
    )

    results: Dict[str, List[Dict]] = {"dqn": [], **{b: [] for b in ALL_BASELINES}}

    for s in series_list:
        # DQN.
        try:
            dqn_summary = _run_dqn(s, checkpoint_dir=Path(checkpoint_dir),
                                   cost=cost, episode_days=episode_days)
            results["dqn"].append(dqn_summary)
        except Exception as e:
            log.warning("inventory.eval.dqn_failed", extra={"error": str(e)})

        # Baselines.
        for b in ALL_BASELINES:
            try:
                r = run_baseline(b, s, cost=cost, episode_days=episode_days, seed=seed)
                results[b].append(r.to_dict())
            except Exception as e:
                log.warning("inventory.eval.baseline_failed", extra={"baseline": b, "error": str(e)})

    # Aggregate.
    aggregate = {}
    for policy, rows in results.items():
        if not rows:
            continue
        aggregate[policy] = {
            "n_series": len(rows),
            "mean_total_cost": float(np.mean([r["total_cost"] for r in rows])),
            "std_total_cost": float(np.std([r["total_cost"] for r in rows])),
            "mean_service_level": float(np.mean([r["service_level"] for r in rows])),
            "mean_unmet_units": float(np.mean([r["unmet_units"] for r in rows])),
            "mean_n_orders": float(np.mean([r["n_orders"] for r in rows])),
            "mean_avg_inventory": float(np.mean([r["avg_inventory"] for r in rows])),
        }

    payload = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_dir),
        "cost_params": cost.to_dict(),
        "aggregate": aggregate,
        "per_series": results,
    }
    out_path = output_dir / f"inventory_eval_{run_name}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Plot comparison.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        policies = [p for p in aggregate.keys()]
        costs = [aggregate[p]["mean_total_cost"] for p in policies]
        services = [aggregate[p]["mean_service_level"] for p in policies]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.bar(policies, costs)
        ax1.set_title("Mean total cost")
        ax1.set_ylabel("cost")
        ax1.tick_params(axis="x", rotation=20)
        ax2.bar(policies, services)
        ax2.set_title("Mean service level")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        plots_dir = Path("reports/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots_dir / f"inventory_{run_name}_comparison.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        log.warning("inventory.eval.plot_failed", extra={"error": str(e)})

    log.info("inventory.eval.done", extra={"output": str(out_path)})
    return payload


def _main():
    p = argparse.ArgumentParser(description="Evaluate DQN vs baselines.")
    p.add_argument("--features", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="reports/metrics")
    p.add_argument("--run-name", required=True)
    p.add_argument("--max-series", type=int, default=5)
    p.add_argument("--episode-days", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_inventory_eval(
        features_dir=args.features,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output,
        run_name=args.run_name,
        max_series=args.max_series,
        episode_days=args.episode_days,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()