"""Gradient-utility prediction loss: supervises router against gradient-derived targets.

During training, we use a simple proxy: the target utility is approximated by the
magnitude of the gradient of the task loss with respect to each expert output.
This is computed via a one-step stop-gradient approximation.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def gradient_prediction_loss(routing_scores: Tensor, routing_targets: Tensor) -> Tensor:
    """KL-divergence from routing score distribution to gradient-utility targets.

    Args:
        routing_scores: (B, N, E) logits from utility predictor.
        routing_targets: (B, N, E) positive target utilities (from gradient magnitudes).

    Returns:
        Scalar loss.
    """
    log_probs = F.log_softmax(routing_scores, dim=-1)
    target_probs = F.softmax(routing_targets.detach(), dim=-1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")
