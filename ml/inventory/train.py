"""DQN training loop for the inventory policy.

Deterministic given the seed. Uses:
    - ReplayBuffer (ml.inventory.replay_buffer)
    - DQNAgent (ml.inventory.dqn_model)
    - InventoryEnvironment (ml.inventory.environment)

The training series is loaded from the same curated dataset used by the
demand and supplier models, so no artificial demand is injected.

Saves:
    artifacts/inventory/<mode>/dqn_best.pt
    artifacts/inventory/<mode>/dqn_last.pt
    artifacts/inventory/<mode>/training_metadata.json
    artifacts/inventory/<mode>/metrics.json

Also records `EnvConfig` (including cost parameters), action multipliers,
state_dim, and n_actions so inference is exact.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.config import get_settings
from core.deterministic import derive_seed, rng_from_seed, seed_everything
from core.logging import configure_logging, get_logger
from ml.inventory.cost_model import CostParameters
from ml.inventory.dqn_model import DQNAgent, QNetworkConfig
from ml.inventory.environment import EnvConfig, InventoryEnvironment
from ml.inventory.replay_buffer import ReplayBuffer

log = get_logger(__name__)


@dataclass
class DQNTrainConfig:
    features_dir: str
    output_dir: str
    supplier_artifacts_dir: Optional[str] = None   # optional; if None use default risk=0.1
    seed: int = 42
    episode_days: int = 60
    n_episodes: int = 200
    max_series: int = 10
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_steps: int = 2_000
    gamma: float = 0.99
    lr: float = 1e-3
    target_update_steps: int = 500
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    hidden: int = 128
    n_layers: int = 2
    # Cost parameters — one set per product; we take the product's unit cost.
    holding_rate_annual: float = 0.20
    order_fixed_cost: float = 250.0
    shortage_cost_multiplier: float = 3.0
    shipping_cost_per_order: float = 50.0
    device: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Data loading: build one training series per (product, warehouse)
# ---------------------------------------------------------------------------
def _load_series(
    features_dir: Path,
    *,
    supplier_artifacts_dir: Optional[Path],
    max_series: int,
    seed: int,
) -> List[dict]:
    """Return a list of dicts, one per (product, warehouse) with all arrays needed.

    The dict has:
        product_sku, warehouse_code,
        forecast_point, forecast_std,
        actual_demand,
        lead_time_mean, lead_time_std,
        supplier_risk,
        unit_cost
    """
    # Research-scale datasets cannot be loaded wholesale on an 8GB-class
    # workstation. Use the bounded-memory research loader instead.
    if "research" in features_dir.parts:
        from ml.inventory.research_loader import (
            load_research_inventory_series,
        )

        return load_research_inventory_series(
            features_dir,
            input_dir="datasets/raw/research",
            max_series=max_series,
            seed=seed,
        )
    feature_path = next(features_dir.glob("demand_features_*.parquet"), None)
    if feature_path is None:
        raise FileNotFoundError(f"No demand_features_*.parquet in {features_dir}")
    df = pd.read_parquet(feature_path)
    if df.empty:
        raise ValueError(f"Empty demand feature file: {feature_path}")

    # Supplier risk: load if we have a predictor; else use a constant 0.1.
    risk_by_supplier = None
    if supplier_artifacts_dir is not None and supplier_artifacts_dir.exists():
        try:
            from ml.supplier_risk.predict import SupplierRiskPredictor
            predictor = SupplierRiskPredictor(supplier_artifacts_dir)
            # Load supplier features to compute risk per product/date.
            sup_feat_path = next(
                (supplier_artifacts_dir.parent.parent / "curated").glob(
                    "*/features/supplier_features_*.parquet"
                ), None,
            )
            if sup_feat_path is None:
                # Try relative to features_dir.
                sup_feat_path = next(
                    features_dir.glob("supplier_features_*.parquet"), None
                )
            if sup_feat_path is not None and sup_feat_path.exists():
                sup_df = pd.read_parquet(sup_feat_path)
                risk_pred = predictor.predict_dataframe(sup_df)
                # Aggregate risk by product_sku + date (mean).
                risk_pred["date"] = pd.to_datetime(risk_pred["date"])
                risk_by_supplier = (
                    risk_pred.groupby(["product_sku", "date"])["risk_probability"]
                    .mean()
                    .reset_index()
                    .rename(columns={"risk_probability": "supplier_risk"})
                )
        except Exception as e:
            log.warning("inventory.train.supplier_risk_skipped", extra={"error": str(e)})

    # Supplier terms: pull lead-time mean/std from the supplier_products feature table.
    sp_path = next(features_dir.glob("supplier_features_*.parquet"), None)
    sp_summary = None
    if sp_path is not None:
        sp_df = pd.read_parquet(sp_path)
        if not sp_df.empty:
            sp_summary = (
                sp_df.groupby("product_sku")
                .agg(
                    lead_time_mean=("feat_sup_lead_time_days", "mean"),
                    lead_time_std=("feat_sup_lead_time_std_days", "mean"),
                )
                .reset_index()
            )

    # Build per-series dicts.
    series_list = []
    grouped = df.groupby(["product_sku", "warehouse_code"], sort=False)
    keys = list(grouped.groups.keys())[:max_series]
    for sku, wh in keys:
        g = df[(df["product_sku"] == sku) & (df["warehouse_code"] == wh)].sort_values("date")
        T = len(g)
        if T < 30:
            continue

        # Forecast point = one-step-ahead "rolling lag-1" as a proxy if we don't
        # want to run the LSTM during DQN data prep. This is a documented design
        # choice: the DQN consumes the model's *signal* — for the training
        # series we use the actual forecast values that the demand model
        # produced on the training split. In practice we would persist those
        # predictions; here we approximate with lag-1 (a lower bound).
        fc = (
            g["feat_demand_lag_1"]
            .ffill()
            .bfill()
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        # Forecast uncertainty = rolling std of residuals. We approximate with
        # the rolling std feature if present, else 0.15 * fc.
        if "feat_demand_roll_std_28" in g.columns:
            fc_std = g["feat_demand_roll_std_28"].fillna(0.0).to_numpy(dtype=np.float32)
        else:
            fc_std = 0.15 * fc
        actual = g["quantity"].to_numpy(dtype=np.float32)

        # Lead time.
        if sp_summary is not None and (sp_summary["product_sku"] == sku).any():
            ltm = float(sp_summary.loc[sp_summary["product_sku"] == sku, "lead_time_mean"].iloc[0])
            lts = float(sp_summary.loc[sp_summary["product_sku"] == sku, "lead_time_std"].iloc[0])
        else:
            ltm, lts = 7.0, 1.5
        lt_mean = np.full(T, ltm, dtype=np.float32)
        lt_std = np.full(T, lts, dtype=np.float32)

        # Risk.
        if risk_by_supplier is not None:
            risk_series = (
                g[["date"]].merge(
                    risk_by_supplier[risk_by_supplier["product_sku"] == sku],
                    on="date", how="left",
                )
            )
            risk = risk_series["supplier_risk"].fillna(0.1).to_numpy(dtype=np.float32)
        else:
            risk = np.full(T, 0.1, dtype=np.float32)

        unit_cost = float(g["unit_cost"].iloc[0])
        series_list.append({
            "product_sku": sku,
            "warehouse_code": wh,
            "forecast_point": fc,
            "forecast_std": fc_std,
            "actual_demand": actual,
            "lead_time_mean": lt_mean,
            "lead_time_std": lt_std,
            "supplier_risk": risk,
            "unit_cost": unit_cost,
        })
    return series_list


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _epsilon(step: int, cfg: DQNTrainConfig) -> float:
    frac = min(1.0, step / max(1, cfg.eps_decay_steps))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def _initial_inventory_from_series(s: dict) -> float:
    # Start with ~ 30 days of expected demand, so the agent begins from a
    # realistic operating point rather than zero.
    return float(np.mean(s["forecast_point"][:30]) * 30.0)


def train_inventory_dqn(cfg: DQNTrainConfig) -> dict:
    configure_logging()
    settings = get_settings()
    seed_everything(cfg.seed, deterministic=settings.DETERMINISTIC)
    device = _resolve_device(cfg.device)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("inventory.train.start", extra={
        "device": str(device), "seed": cfg.seed, "output": str(out_dir),
    })

    series_list = _load_series(
        Path(cfg.features_dir),
        supplier_artifacts_dir=Path(cfg.supplier_artifacts_dir) if cfg.supplier_artifacts_dir else None,
        max_series=cfg.max_series,
        seed=cfg.seed,
    )
    if not series_list:
        raise ValueError("No usable series for DQN training.")

    # Environments and reward scaling derived from the FIRST series' cost.
    unit_cost = float(series_list[0]["unit_cost"])
    cost = CostParameters(
        unit_cost=unit_cost,
        holding_rate_annual=cfg.holding_rate_annual,
        order_fixed_cost=cfg.order_fixed_cost,
        shortage_cost_per_unit=unit_cost * cfg.shortage_cost_multiplier,
        shipping_cost_per_order=cfg.shipping_cost_per_order,
    )
    env_cfg = EnvConfig(
        cost=cost,
        episode_days=cfg.episode_days,
    )
    env = InventoryEnvironment(env_cfg)

    # Action space.
    action_multipliers = env.cfg.action_multipliers
    n_actions = len(action_multipliers)
    state_dim = 11

    # Networks.
    q_cfg = QNetworkConfig(
        state_dim=state_dim, n_actions=n_actions,
        hidden=cfg.hidden, n_layers=cfg.n_layers,
    )
    agent = DQNAgent(q_cfg).to(device)
    optim = torch.optim.Adam(agent.online.parameters(), lr=cfg.lr)
    loss_fn = nn.SmoothL1Loss()

    buffer = ReplayBuffer(cfg.replay_capacity, state_dim)
    rng = rng_from_seed(derive_seed(cfg.seed, "dqn_replay"))

    # Training.
    global_step = 0
    best_eval = math.inf
    history: List[dict] = []

    for ep in range(cfg.n_episodes):
        # Pick a series (round-robin so every series gets exercised).
        s = series_list[ep % len(series_list)]
        env.attach_series(
            forecast_point=s["forecast_point"],
            forecast_std=s["forecast_std"],
            actual_demand=s["actual_demand"],
            lead_time_mean=s["lead_time_mean"],
            lead_time_std=s["lead_time_std"],
            supplier_risk=s["supplier_risk"],
            initial_inventory=_initial_inventory_from_series(s),
            initial_inbound=0.0,
            service_target=0.95,
            seed=derive_seed(cfg.seed, "env", ep),
        )
        state = env.reset(start_index=0)

        # Ensure episode fits within series length.
        max_start = max(1, len(s["actual_demand"]) - cfg.episode_days - 2)
        start = int(rng.integers(0, max_start)) if max_start > 1 else 0
        state = env.reset(start_index=start)

        ep_reward = 0.0
        ep_loss_sum = 0.0
        ep_loss_count = 0

        done = False
        while not done:
            eps = _epsilon(global_step, cfg)
            if rng.random() < eps or len(buffer) < cfg.warmup_steps:
                action = int(rng.integers(0, n_actions))
            else:
                with torch.no_grad():
                    q = agent.q_values(torch.from_numpy(state[None, :]).to(device))
                    action = int(q.argmax(dim=1).item())

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            global_step += 1

            # Train.
            if len(buffer) >= max(cfg.warmup_steps, cfg.batch_size):
                b_s, b_a, b_r, b_ns, b_d = buffer.sample(cfg.batch_size, rng)
                b_s_t = torch.from_numpy(b_s).to(device)
                b_a_t = torch.from_numpy(b_a).to(device)
                b_r_t = torch.from_numpy(b_r).to(device)
                b_ns_t = torch.from_numpy(b_ns).to(device)
                b_d_t = torch.from_numpy(b_d).to(device)

                q_all = agent.online(b_s_t)
                q_taken = q_all.gather(1, b_a_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = agent.target_q_values(b_ns_t).max(dim=1).values
                    target = b_r_t + cfg.gamma * (1.0 - b_d_t) * q_next

                loss = loss_fn(q_taken, target)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(agent.online.parameters(), cfg.grad_clip)
                optim.step()
                ep_loss_sum += float(loss.detach().cpu())
                ep_loss_count += 1

                if global_step % cfg.target_update_steps == 0:
                    agent.sync_target()

        summary = env.episode_summary()
        summary["episode"] = ep
        summary["epsilon"] = float(_epsilon(global_step, cfg))
        summary["mean_loss"] = float(ep_loss_sum / max(ep_loss_count, 1))
        history.append(summary)

        if (ep + 1) % 10 == 0 or ep == 0:
            log.info("inventory.train.episode", extra={
                "episode": ep,
                "episode_reward": summary["reward"],
                "total_cost": summary["total_cost"],
                "service_level": summary["service_level"],
                "epsilon": summary["epsilon"],
            })

        # Deterministic eval on a fixed series every 20 episodes.
        if (ep + 1) % 20 == 0:
            eval_cost = _deterministic_eval(agent, series_list, cfg, device)
            if eval_cost < best_eval:
                best_eval = eval_cost
                agent.save(str(out_dir / "dqn_best.pt"))
                log.info("inventory.train.best", extra={"episode": ep, "eval_cost": eval_cost})

        # Always save last.
        agent.save(str(out_dir / "dqn_last.pt"))

    # Final deterministic eval.
    final_eval = _deterministic_eval(agent, series_list, cfg, device)

    metadata = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "seed": cfg.seed,
        "device": str(device),
        "state_dim": state_dim,
        "n_actions": n_actions,
        "action_multipliers": list(action_multipliers),
        "env_config": env_cfg.to_dict(),
        "q_network_config": q_cfg.to_dict(),
        "train_config": cfg.to_dict(),
        "best_eval_cost": best_eval,
        "final_eval_cost": final_eval,
        "history": history,
        "n_series": len(series_list),
    }
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps({"best_eval_cost": best_eval, "final_eval_cost": final_eval}, indent=2),
        encoding="utf-8",
    )

    log.info("inventory.train.done", extra={
        "best_eval_cost": best_eval, "final_eval_cost": final_eval,
    })
    return metadata


@torch.no_grad()
def _deterministic_eval(
    agent: DQNAgent,
    series_list: List[dict],
    cfg: DQNTrainConfig,
    device: torch.device,
) -> float:
    """Evaluate greedily across a subset of series; return mean total cost."""
    agent.eval()
    total_cost = 0.0
    n = 0
    cost = CostParameters(
        unit_cost=float(series_list[0]["unit_cost"]),
        holding_rate_annual=cfg.holding_rate_annual,
        order_fixed_cost=cfg.order_fixed_cost,
        shortage_cost_per_unit=float(series_list[0]["unit_cost"]) * cfg.shortage_cost_multiplier,
        shipping_cost_per_order=cfg.shipping_cost_per_order,
    )
    env_cfg = EnvConfig(cost=cost, episode_days=cfg.episode_days)
    env = InventoryEnvironment(env_cfg)

    for s in series_list[: min(3, len(series_list))]:
        env.attach_series(
            forecast_point=s["forecast_point"],
            forecast_std=s["forecast_std"],
            actual_demand=s["actual_demand"],
            lead_time_mean=s["lead_time_mean"],
            lead_time_std=s["lead_time_std"],
            supplier_risk=s["supplier_risk"],
            initial_inventory=_initial_inventory_from_series(s),
            seed=cfg.seed,
        )
        state = env.reset(start_index=0)
        done = False
        while not done:
            q = agent.q_values(torch.from_numpy(state[None, :]).to(device))
            a = int(q.argmax(dim=1).item())
            state, _r, done, _ = env.step(a)
        total_cost += env.episode_summary()["total_cost"]
        n += 1

    agent.train()
    return float(total_cost / max(n, 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> DQNTrainConfig:
    p = argparse.ArgumentParser(description="Train the DQN inventory policy.")
    p.add_argument("--features", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--supplier-artifacts", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode-days", type=int, default=60)
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--max-series", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-capacity", type=int, default=50_000)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target-update-steps", type=int, default=500)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    return DQNTrainConfig(
        features_dir=args.features,
        output_dir=args.output,
        supplier_artifacts_dir=args.supplier_artifacts,
        seed=args.seed,
        episode_days=args.episode_days,
        n_episodes=args.n_episodes,
        max_series=args.max_series,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        gamma=args.gamma,
        target_update_steps=args.target_update_steps,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    result = train_inventory_dqn(cfg)
    print(json.dumps({
        "best_eval_cost": result["best_eval_cost"],
        "final_eval_cost": result["final_eval_cost"],
    }, indent=2))
