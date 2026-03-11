"""GRAFT Block tests."""

import torch

from graft_net.models.block import GraftBlock
from graft_net.models.config import GraftNetConfig


def test_graft_block_returns_residual_shaped_output() -> None:
    cfg = GraftNetConfig(embed_dim=32, num_heads=4, num_experts=4,
                         topology_edge_hidden=16, expert_hidden_dim=64, predictor_hidden_dim=32)
    block = GraftBlock(cfg)
    x = torch.randn(2, 8, 32)
    out = block(x)
    assert out.hidden.shape == x.shape


def test_graft_block_ablation_all_off_still_runs() -> None:
    cfg = GraftNetConfig(
        embed_dim=32, num_heads=4, num_experts=4,
        topology_edge_hidden=16, expert_hidden_dim=64, predictor_hidden_dim=32,
        use_predictive_attention=False,
        use_latent_topology=False,
        use_gradient_routing=False,
    )
    block = GraftBlock(cfg)
    x = torch.randn(2, 8, 32)
    out = block(x)
    assert out.hidden.shape == x.shape
