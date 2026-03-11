"""Graph message passing using the sparsified adjacency matrix."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GraphMessagePassing(nn.Module):
    """One-hop message aggregation with linear projection."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.msg_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x:   (B, N, D) node features.
            adj: (B, N, N) adjacency mask (bool or float).

        Returns:
            agg: (B, N, D) aggregated node representations.
        """
        msgs = self.msg_proj(x)             # (B, N, D)
        # adj: (B, N, N) — adj[b, i, j]=True means i receives from j
        # Weighted average of incoming messages
        adj_float = adj.float()
        deg = adj_float.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, N, 1)
        agg = torch.bmm(adj_float, msgs) / deg                    # (B, N, D)
        return self.out_proj(agg)
