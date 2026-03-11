"""Future-state prediction loss: MSE between predicted and target future representations."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def future_prediction_loss(future_state: Tensor, target: Tensor) -> Tensor:
    """MSE loss between predicted future state and target representation.

    Args:
        future_state: (B, N, D) predictor output.
        target: (B, N, D) target (e.g. next-layer hidden state, stop-gradiented).

    Returns:
        Scalar loss.
    """
    return F.mse_loss(future_state, target.detach())
