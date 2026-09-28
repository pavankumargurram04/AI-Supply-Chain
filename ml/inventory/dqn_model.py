"""Q-network and target network.

A small MLP with two hidden layers. Two outputs (no dueling, no
distributional head) to keep the implementation transparent.

Target network is a copy of the online network. Hard update every
`target_update_steps` training steps (done by the training loop, not here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class QNetworkConfig:
    state_dim: int
    n_actions: int
    hidden: int = 128
    n_layers: int = 2
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
        }


class QNetwork(nn.Module):
    def __init__(self, cfg: QNetworkConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers = []
        in_dim = cfg.state_dim
        for _ in range(cfg.n_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.hidden
        layers.append(nn.Linear(in_dim, cfg.n_actions))
        self.net = nn.Sequential(*layers)

        # Small init for stability.
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(nn.Module):
    """Container for online + target networks. Stateless apart from params."""

    def __init__(self, cfg: QNetworkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.online = QNetwork(cfg)
        self.target = QNetwork(cfg)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.online(x)

    @torch.no_grad()
    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.online(x)

    @torch.no_grad()
    def target_q_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.target(x)

    @torch.no_grad()
    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str) -> None:
        torch.save({
            "config": self.cfg.to_dict(),
            "online_state": self.online.state_dict(),
            "target_state": self.target.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, *, device: Optional[torch.device] = None) -> "DQNAgent":
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls(QNetworkConfig(**ckpt["config"]))
        agent.online.load_state_dict(ckpt["online_state"])
        agent.target.load_state_dict(ckpt["target_state"])
        if device is not None:
            agent.to(device)
        return agent