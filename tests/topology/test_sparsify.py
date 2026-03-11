"""Topology sparsification tests."""

import torch

from graft_net.topology.sparsify import topk_adjacency


def test_topk_adjacency_keeps_at_most_k_edges_per_node() -> None:
    scores = torch.tensor([[[0.1, 0.5, 0.4], [0.2, 0.3, 0.9], [0.8, 0.7, 0.1]]])
    adj = topk_adjacency(scores, k=1)
    assert int(adj[0, 0].sum().item()) == 1
    assert int(adj[0, 1].sum().item()) == 1
    assert int(adj[0, 2].sum().item()) == 1


def test_topk_adjacency_selects_highest_scores() -> None:
    scores = torch.tensor([[[0.0, 0.0, 1.0]]])  # (1, 1, 3)
    adj = topk_adjacency(scores, k=1)
    assert adj[0, 0, 2].item() is True
    assert adj[0, 0, 0].item() is False


def test_topk_adjacency_clamps_k_to_n() -> None:
    scores = torch.randn(2, 4, 4)
    adj = topk_adjacency(scores, k=10)  # k > n — should not raise
    assert adj.shape == (2, 4, 4)
