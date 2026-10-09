"""Predictive Attention module.

Queries are derived from a predicted future latent state rather than the current state.
This is GRAFT-Net's core attention hypothesis (H1).

Ablation path: when use_predictive_attention=False, queries derive from current state
(standard multi-head attention) and future_state = x (identity short-circuit).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graft_net.models.config import GraftNetConfig


@dataclass
class PredictiveAttentionOutput:
    attended: Tensor  # (B, N, D) output after attention
    future_state: Tensor  # (B, N, D) predicted future representation
    attention_weights: Tensor  # (B, H, N, N) attention probabilities


class PredictiveAttention(nn.Module):
    """Multi-head attention with future-state query prediction."""

    def __init__(self, cfg: GraftNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        h = cfg.num_heads
        assert d % h == 0, f"embed_dim {d} must be divisible by num_heads {h}"
        self.head_dim = d // h

        # Future-state predictor MLP
        self.predictor = nn.Sequential(
            nn.Linear(d, cfg.predictor_hidden_dim),
            nn.GELU(),
            *[
                layer
                for _ in range(cfg.predictor_depth - 1)
                for layer in (
                    nn.Linear(cfg.predictor_hidden_dim, cfg.predictor_hidden_dim),
                    nn.GELU(),
                )
            ],
            nn.Linear(cfg.predictor_hidden_dim, d),
        )

        # Q from future state, K+V from current state
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> PredictiveAttentionOutput:
        b, n, d = x.shape
        h = self.cfg.num_heads
        dh = self.head_dim

        if self.cfg.use_predictive_attention:
            future_state = self.predictor(x)  # (B, N, D)
            q_src = future_state
        else:
            future_state = x  # bypass: identity
            q_src = x

        q = self.q_proj(q_src).view(b, n, h, dh).transpose(1, 2)  # (B, H, N, dh)
        k = self.k_proj(x).view(b, n, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(b, n, h, dh).transpose(1, 2)

        scale = dh**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)

        if attention_mask is not None:
            # attention_mask: (B, N) True = keep
            mask_4d = attention_mask[:, None, None, :].expand_as(scores)
            scores = scores.masked_fill(~mask_4d, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attended = torch.matmul(weights, v)  # (B, H, N, dh)
        attended = attended.transpose(1, 2).contiguous().view(b, n, d)
        attended = self.out_proj(attended)

        return PredictiveAttentionOutput(
            attended=attended,
            future_state=future_state,
            attention_weights=weights,
        )
