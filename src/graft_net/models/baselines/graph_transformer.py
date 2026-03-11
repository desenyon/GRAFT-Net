"""Graph Transformer baseline with explicit message passing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class GraphTransformerOutput:
    hidden: Tensor


class GraphTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim),
        )
        self.msg_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, adj: Tensor | None = None) -> Tensor:
        nx = self.norm1(x)
        attn_out, _ = self.attn(nx, nx, nx)
        x = x + self.drop(attn_out)

        if adj is not None:
            msgs = self.msg_proj(x)
            adj_f = adj.float()
            deg = adj_f.sum(-1, keepdim=True).clamp(min=1.0)
            graph_msg = torch.bmm(adj_f, msgs) / deg
            gate = self.gate(torch.cat([x, graph_msg], dim=-1))
            x = x + self.drop(gate * graph_msg)

        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class GraphTransformer(nn.Module):
    """Transformer with graph-aware message passing (explicit adjacency input)."""

    def __init__(self, embed_dim: int = 256, num_layers: int = 4, num_heads: int = 8,
                 ffn_hidden: int = 512, dropout: float = 0.1, max_seq_len: int = 512) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(embed_dim, num_heads, ffn_hidden, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, adjacency: Tensor | None = None, **_: object) -> GraphTransformerOutput:
        b, n, d = x.shape
        positions = torch.arange(n, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x, adj=adjacency)
        return GraphTransformerOutput(hidden=self.norm(x))
