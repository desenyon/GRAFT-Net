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


def test_legacy_registry_default_ffn_width_tracks_embedding_width():
    transformer = build_model("transformer", embed_dim=32, num_layers=1, num_heads=4)
    graph = build_model("graph_transformer", embed_dim=32, num_layers=1, num_heads=4)
    moe = build_model("moe_transformer", embed_dim=32, num_layers=1, num_heads=4)
    assert transformer.encoder.layers[0].linear1.out_features == 64
    assert graph.blocks[0].ffn[0].out_features == 64
    assert moe.blocks[0].moe.experts[0][0].out_features == 64
