"""Topology regularisation losses: sparsity + entropy penalties."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def topology_loss(soft_adjacency: Tensor) -> Tensor:
    """Encourage sparse, informative graphs.

    Combines:
    - L1 sparsity on soft adjacency
    - Entropy term: penalise degenerate uniform or collapsed distributions

    Args:
        soft_adjacency: (B, N, N) softmax-normalised edge weights.

    Returns:
        Scalar regularisation loss.
    """
    # L1 sparsity: push small edges toward zero
    l1 = soft_adjacency.abs().mean()

    # Negative entropy: maximise diversity of edge distributions (prevent collapse)
    eps = 1e-8
    H = -(soft_adjacency * (soft_adjacency + eps).log()).sum(-1).mean()
    neg_entropy = -H  # positive when entropy is low (encourage high entropy)

    return l1 + 0.1 * neg_entropy
