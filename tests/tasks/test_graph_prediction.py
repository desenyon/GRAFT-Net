"""Graph prediction task tests (EXPERIMENTAL)."""

import torch

from graft_net.tasks.graph_prediction import GraphPredictionModel


def test_graph_prediction_model_returns_graph_logits() -> None:
    model = GraphPredictionModel(embed_dim=32, num_layers=2, num_heads=4, num_classes=2)
    batch = {
        "node_features": torch.randn(2, 10, 32),
        "adjacency": torch.randint(0, 2, (2, 10, 10)).float(),
        "labels": torch.tensor([0, 1]),
    }
    out = model(batch)
    assert out["logits"].shape == (2, 2)
    assert out["task_loss"].item() >= 0
