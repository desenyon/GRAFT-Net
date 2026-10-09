"""Shared tensor utility functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Softmax that zeroes out masked positions.

    Args:
        logits: Raw scores of any shape.
        mask: Boolean tensor; True = keep, False = mask out.
        dim: Dimension along which to apply softmax.

    Returns:
        Probabilities with masked positions set to zero.
    """
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = F.softmax(logits, dim=dim)
    probs = probs * mask
    return probs / probs.sum(dim=dim, keepdim=True).clamp(min=torch.finfo(probs.dtype).eps)


def pairwise_features(x: Tensor) -> Tensor:
    """Compute pairwise concatenated features between all token pairs.

    Args:
        x: (B, N, D) token representations.

    Returns:
        (B, N, N, 2D) pairwise features.
    """
    b, n, d = x.shape
    xi = x.unsqueeze(2).expand(b, n, n, d)  # (B, N, N, D)
    xj = x.unsqueeze(1).expand(b, n, n, d)  # (B, N, N, D)
    return torch.cat([xi, xj], dim=-1)  # (B, N, N, 2D)


def gelu_approx(x: Tensor) -> Tensor:
    """Fast GELU approximation."""
    return F.gelu(x, approximate="tanh")
