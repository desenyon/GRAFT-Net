"""Gradient router tests."""

import torch

from graft_net.routing.gradient_router import topk_route


def test_topk_route_selects_highest_scoring_experts() -> None:
    scores = torch.tensor([[[0.1, 0.7, 0.2]]])
    weights, indices = topk_route(scores, k=1)
    assert indices.tolist() == [[[1]]]
    assert weights.shape == (1, 1, 1)


def test_topk_route_weights_sum_to_one() -> None:
    scores = torch.randn(2, 8, 6)
    weights, _indices = topk_route(scores, k=3)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 8))


def test_topk_route_respects_k() -> None:
    scores = torch.randn(1, 5, 10)
    _, indices = topk_route(scores, k=4)
    assert indices.shape == (1, 5, 4)
