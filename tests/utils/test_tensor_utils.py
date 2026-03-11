"""Tensor utility tests."""

import torch

from graft_net.utils.tensor import masked_softmax, pairwise_features


def test_masked_softmax_zeroes_masked_positions() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    probs = masked_softmax(logits, mask, dim=-1)
    assert probs[0, 1].item() == 0.0
    assert torch.isclose(probs[0, 0] + probs[0, 2], torch.tensor(1.0))


def test_masked_softmax_full_mask_returns_uniform_or_zero() -> None:
    logits = torch.ones(1, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)
    probs = masked_softmax(logits, mask)
    assert torch.allclose(probs, torch.full_like(probs, 0.25))


def test_pairwise_features_shape() -> None:
    x = torch.randn(2, 5, 16)
    pairs = pairwise_features(x)
    assert pairs.shape == (2, 5, 5, 32)
