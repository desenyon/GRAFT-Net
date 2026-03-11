"""Baseline model registry tests."""

import torch

from graft_net.models.registry import build_model


def test_build_model_supports_all_baseline_names() -> None:
    for name in ["graft_net", "transformer", "moe_transformer", "graph_transformer"]:
        model = build_model(name=name, embed_dim=32, num_layers=2, num_heads=4)
        x = torch.randn(2, 8, 32)
        out = model(x)
        assert out.hidden.shape == (2, 8, 32), f"{name} output shape mismatch"


def test_build_model_raises_for_unknown() -> None:
    import pytest
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model("not_a_model")
