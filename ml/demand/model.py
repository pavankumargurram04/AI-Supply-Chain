"""Global LSTM demand model.

Architecture:
    encoder:  LSTM over (L, n_enc) → h_enc  (last hidden state)
    static:   Embeddings (product, category, warehouse) + numeric static vector
    decoder:  for each of H future steps, given (h_dec_{h-1}, h_enc, dec_h),
              advance an LSTM cell and produce a scalar prediction.

This is a standard seq2seq forecasting head that respects the "known future"
(calendar) versus "unknown future" (demand) distinction.

Optional quantile head:
    Set `quantiles=(0.1, 0.5, 0.9)` and the final projection outputs
    `len(quantiles)` values per horizon step. The training loss then switches
    to pinball loss. Otherwise the model predicts a single point per step and
    training uses MSE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class DemandModelConfig:
    n_products: int
    n_categories: int
    n_warehouses: int
    n_encoder_features: int
    n_decoder_features: int
    n_static_num: int

    horizon: int = 30

    # Embedding sizes.
    product_emb_dim: int = 32
    category_emb_dim: int = 8
    warehouse_emb_dim: int = 8

    # Hidden sizes.
    encoder_hidden: int = 128
    encoder_layers: int = 1
    decoder_hidden: int = 128
    static_mlp_hidden: int = 64
    decoder_mlp_hidden: int = 64

    dropout: float = 0.10

    # Optional quantile head. If None, point forecast (MSE loss).
    quantiles: Optional[Tuple[float, ...]] = None

    def to_dict(self) -> dict:
        return {
            "n_products": self.n_products,
            "n_categories": self.n_categories,
            "n_warehouses": self.n_warehouses,
            "n_encoder_features": self.n_encoder_features,
            "n_decoder_features": self.n_decoder_features,
            "n_static_num": self.n_static_num,
            "horizon": self.horizon,
            "product_emb_dim": self.product_emb_dim,
            "category_emb_dim": self.category_emb_dim,
            "warehouse_emb_dim": self.warehouse_emb_dim,
            "encoder_hidden": self.encoder_hidden,
            "encoder_layers": self.encoder_layers,
            "decoder_hidden": self.decoder_hidden,
            "static_mlp_hidden": self.static_mlp_hidden,
            "decoder_mlp_hidden": self.decoder_mlp_hidden,
            "dropout": self.dropout,
            "quantiles": list(self.quantiles) if self.quantiles else None,
        }


class GlobalDemandLSTM(nn.Module):
    def __init__(self, cfg: DemandModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # --- Embeddings ---
        self.product_emb = nn.Embedding(cfg.n_products, cfg.product_emb_dim)
        self.category_emb = nn.Embedding(cfg.n_categories, cfg.category_emb_dim)
        self.warehouse_emb = nn.Embedding(cfg.n_warehouses, cfg.warehouse_emb_dim)
        nn.init.normal_(self.product_emb.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.category_emb.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.warehouse_emb.weight, mean=0.0, std=0.05)

        static_dim = (
            cfg.product_emb_dim
            + cfg.category_emb_dim
            + cfg.warehouse_emb_dim
            + cfg.n_static_num
        )

        # --- Static MLP ---
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, cfg.static_mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.static_mlp_hidden, cfg.static_mlp_hidden),
            nn.ReLU(),
        )

        # --- Encoder ---
        self.encoder = nn.LSTM(
            input_size=cfg.n_encoder_features,
            hidden_size=cfg.encoder_hidden,
            num_layers=cfg.encoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.encoder_layers > 1 else 0.0,
        )
        self.encoder_drop = nn.Dropout(cfg.dropout)

        # --- Decoder ---
        # At each future step we feed: [prev_h_dec, h_enc, static_repr, dec_t]
        dec_input_dim = cfg.decoder_hidden + cfg.encoder_hidden + cfg.static_mlp_hidden + cfg.n_decoder_features
        self.decoder_cell = nn.LSTMCell(dec_input_dim, cfg.decoder_hidden)

        # Init the initial hidden state for the decoder from encoder + static.
        self.h0_proj = nn.Linear(cfg.encoder_hidden + cfg.static_mlp_hidden, cfg.decoder_hidden)
        self.c0_proj = nn.Linear(cfg.encoder_hidden + cfg.static_mlp_hidden, cfg.decoder_hidden)

        # --- Output head ---
        out_dim = len(cfg.quantiles) if cfg.quantiles else 1
        self.head = nn.Sequential(
            nn.Linear(cfg.decoder_hidden + cfg.n_decoder_features, cfg.decoder_mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.decoder_mlp_hidden, out_dim),
        )

    # -- Forward --------------------------------------------------------
    def forward(
        self,
        encoder: torch.Tensor,      # (B, L, n_enc)
        decoder: torch.Tensor,      # (B, H, n_dec)
        static_cat: torch.Tensor,   # (B, 3) — [prod_idx, cat_idx, wh_idx]
        static_num: torch.Tensor,   # (B, n_static_num)
    ) -> torch.Tensor:
        B, H, _ = decoder.shape

        # Embeddings
        p = self.product_emb(static_cat[:, 0])
        c = self.category_emb(static_cat[:, 1])
        w = self.warehouse_emb(static_cat[:, 2])
        static_repr = self.static_mlp(torch.cat([p, c, w, static_num], dim=-1))  # (B, S)

        # Encoder
        enc_out, (h_n, _) = self.encoder(encoder)  # h_n: (layers, B, enc_hidden)
        h_enc = h_n[-1]                            # (B, enc_hidden)
        h_enc = self.encoder_drop(h_enc)

        # Initialise decoder states.
        cat = torch.cat([h_enc, static_repr], dim=-1)
        h_dec = torch.tanh(self.h0_proj(cat))
        c_dec = torch.tanh(self.c0_proj(cat))

        outputs = []
        prev_pred = torch.zeros(B, 1, device=encoder.device)

        for t in range(H):
            dec_t = decoder[:, t, :]                            # (B, n_dec)
            cell_in = torch.cat([h_dec, h_enc, static_repr, dec_t], dim=-1)
            h_dec, c_dec = self.decoder_cell(cell_in, (h_dec, c_dec))
            head_in = torch.cat([h_dec, dec_t], dim=-1)
            y = self.head(head_in)                              # (B, 1) or (B, Q)
            outputs.append(y)

        stacked = torch.stack(outputs, dim=1)                   # (B, H, 1) or (B, H, Q)
        if self.cfg.quantiles is None:
            return stacked.squeeze(-1)                          # (B, H)
        return stacked                                          # (B, H, Q)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def pinball_loss(
    pred: torch.Tensor,       # (B, H, Q)
    target: torch.Tensor,     # (B, H)
    quantiles: Sequence[float],
) -> torch.Tensor:
    """Quantile (pinball) loss. Mean over batch, horizon, and quantiles."""
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * err, (q - 1.0) * err)
    return loss.mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)