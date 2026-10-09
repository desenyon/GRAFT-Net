"""Topology edge scoring: pairwise dot-product / MLP scoring."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from graft_net.utils.tensor import pairwise_features


class MLPEdgeScorer(nn.Module):
    """Pairwise MLP edge scorer producing soft adjacency logits."""

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            scores: (B, N, N) unnormalised edge logits
        """
        pairs = pairwise_features(x)  # (B, N, N, 2D)
        scores = self.net(pairs).squeeze(-1)  # (B, N, N)
        return scores
