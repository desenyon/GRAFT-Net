"""Load-balance loss: prevent expert collapse in MoE routing."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def load_balance_loss(routing_scores: Tensor) -> Tensor:
    """Encourage uniform expert utilisation.

    Uses the auxiliary loss from Switch Transformer:
    L_balance = num_experts * sum_i(f_i * P_i)
    where f_i = fraction of tokens routed to expert i,
          P_i = mean routing probability for expert i.

    Args:
        routing_scores: (B, N, E) unnormalised logits.

    Returns:
        Scalar load-balance penalty.
    """
    probs = F.softmax(routing_scores, dim=-1)  # (B, N, E)
    _b, _n, e = probs.shape
    # Fraction of tokens selecting each expert (via argmax)
    expert_idx = probs.argmax(-1)  # (B, N)
    one_hot = F.one_hot(expert_idx, num_classes=e).float()  # (B, N, E)
    f_i = one_hot.mean(dim=(0, 1))  # (E,)
    p_i = probs.mean(dim=(0, 1))  # (E,)
    return e * (f_i * p_i).sum()
