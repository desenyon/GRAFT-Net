"""Per-token KL from utility targets to router probabilities.

Task wrappers estimate signed first-order utility using candidate expert outputs
and the task gradient at the mixture output. This function also accepts external
utility logits; targets are always detached before normalization.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def gradient_prediction_loss(routing_scores: Tensor, routing_targets: Tensor) -> Tensor:
    """KL-divergence from routing score distribution to gradient-utility targets.

    Args:
        routing_scores: (B, N, E) logits from utility predictor.
        routing_targets: (B, N, E) target utility logits (larger means more useful).

    Returns:
        Scalar loss.
    """
    log_probs = F.log_softmax(routing_scores, dim=-1)
    target_probs = F.softmax(routing_targets.detach(), dim=-1)
    return F.kl_div(log_probs, target_probs, reduction="none").sum(-1).mean()
