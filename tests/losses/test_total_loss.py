"""Total loss tests."""

import torch

from graft_net.losses.total import compute_total_loss


def test_total_loss_returns_named_components() -> None:
    outputs = {
        "task_loss": torch.tensor(1.0),
        "future_state": torch.zeros(2, 3, 4),
        "future_target": torch.ones(2, 3, 4),
        "routing_scores": torch.zeros(2, 3, 5),
        "routing_targets": torch.ones(2, 3, 5),
        "soft_adjacency": torch.full((2, 3, 3), 1.0 / 3),
        "expert_probs": torch.full((2, 3, 5), 0.2),
    }
    loss = compute_total_loss(outputs)
    assert "total" in loss
    assert "task" in loss
    assert "future" in loss
    assert loss["total"].item() > 0


def test_total_loss_total_is_weighted_sum() -> None:
    outputs = {
        "task_loss": torch.tensor(2.0),
        "future_state": torch.zeros(1, 2, 4),
        "future_target": torch.zeros(1, 2, 4),
        "routing_scores": torch.zeros(1, 2, 3),
        "routing_targets": torch.zeros(1, 2, 3),
        "soft_adjacency": torch.full((1, 2, 2), 0.5),
    }
    loss = compute_total_loss(outputs, lambda_future=0.0, lambda_grad=0.0,
                               lambda_topology=0.0, lambda_balance=0.0)
    assert torch.isclose(loss["total"], torch.tensor(2.0))
