"""Top-k sparsification of soft adjacency matrices."""

from __future__ import annotations

import torch
from torch import Tensor


def topk_adjacency(scores: Tensor, k: int) -> Tensor:
    """Keep the top-k highest-scoring outgoing edges per node.

    Args:
        scores: (B, N, N) edge logit scores — scores[b, i, j] = edge from i to j.
        k: Number of edges to keep per source node.

    Returns:
        binary_adj: (B, N, N) boolean mask, True where edge is kept.
    """
    b, n, m = scores.shape
    k = min(k, m)

    # topk along last dim: for each source node i, keep top-k targets j
    _, top_indices = scores.topk(k, dim=-1)  # (B, N, k)

    binary_adj = torch.zeros(b, n, m, dtype=torch.bool, device=scores.device)
    binary_adj.scatter_(-1, top_indices, True)
    return binary_adj
