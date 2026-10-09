"""Entropy regularization of learned edge distributions."""

from torch import Tensor


def topology_loss(soft_adjacency: Tensor) -> Tensor:
    """Minimize row entropy to encourage concentration; hard top-k fixes degree.

    L1 of a row-stochastic matrix is constant and cannot induce sparsity.
    Zero-probability masked entries contribute zero. This penalty can favor
    collapsed graphs; it is an optional research objective, not a guarantee
    of informative relationships.
    """
    probabilities = soft_adjacency.float()
    return -(probabilities * probabilities.clamp(min=1e-8).log()).sum(-1).mean()
