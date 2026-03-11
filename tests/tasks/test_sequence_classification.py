"""Sequence classification task tests."""

import torch

from graft_net.tasks.sequence_classification import SequenceClassificationModel


def test_sequence_classification_model_produces_logits() -> None:
    model = SequenceClassificationModel(embed_dim=32, num_layers=2, num_heads=4, num_classes=3)
    batch = {"inputs": torch.randn(2, 12, 32), "labels": torch.tensor([0, 1])}
    out = model(batch)
    assert out["logits"].shape == (2, 3)
    assert out["task_loss"].item() >= 0


def test_sequence_classification_backward_passes() -> None:
    model = SequenceClassificationModel(embed_dim=32, num_layers=2, num_heads=4, num_classes=2)
    batch = {"inputs": torch.randn(2, 8, 32), "labels": torch.tensor([0, 1])}
    out = model(batch)
    out["task_loss"].backward()
    # Check at least one parameter has a gradient
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
