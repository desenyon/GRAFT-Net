"""Gradient routing: top-k selection from utility scores."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def topk_route(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Select the top-k experts per token by utility score.

    Args:
        scores: (B, N, E) unnormalised utility scores per expert.
        k: Number of experts to select.

    Returns:
        weights: (B, N, k) normalised weights for selected experts.
        indices: (B, N, k) indices of selected experts.
    """
    k = min(k, scores.shape[-1])
    top_scores, indices = scores.topk(k, dim=-1)  # (B, N, k)
    weights = F.softmax(top_scores, dim=-1)  # (B, N, k)
    return weights, indices
