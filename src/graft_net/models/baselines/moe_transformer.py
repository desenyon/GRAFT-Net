"""MoE Transformer baseline (standard routing, no gradient supervision)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MoETransformerOutput:
    hidden: Tensor


class _MoEFFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        experts_topk: int,
        expert_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.k = experts_topk
        self.router = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, embed_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        _b, _n, _d = x.shape
        scores = self.router(x)  # (B, N, E)
        topk_scores, topk_idx = scores.topk(self.k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)  # (B, N, k)
        out = torch.zeros_like(x)
        for ei, expert in enumerate(self.experts):
            mask = (topk_idx == ei).float()  # (B, N, k)
            w = (weights * mask).sum(-1, keepdim=True)  # (B, N, 1)
            out += w * expert(x)
        return out


class MoETransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        experts_topk: int,
        expert_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.moe = _MoEFFN(embed_dim, num_experts, experts_topk, expert_hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        nx = self.norm1(x)
        attn_out, _ = self.attn(
            nx,
            nx,
            nx,
            key_padding_mask=~attention_mask if attention_mask is not None else None,
            need_weights=False,
        )
        x = x + self.drop(attn_out)
        x = x + self.drop(self.moe(self.norm2(x)))
        return x


class MoETransformer(nn.Module):
    """Transformer with standard similarity-based MoE routing (no gradient supervision)."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_experts: int = 8,
        experts_topk: int = 2,
        expert_hidden: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList(
            [
                MoETransformerBlock(
                    embed_dim, num_heads, num_experts, experts_topk, expert_hidden, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: Tensor, attention_mask: Tensor | None = None, **_: object
    ) -> MoETransformerOutput:
        _b, n, _d = x.shape
        positions = torch.arange(n, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return MoETransformerOutput(hidden=self.norm(x))
