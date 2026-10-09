"""Expert module tests."""

import torch

from graft_net.models.config import GraftNetConfig
from graft_net.modules.experts import GradientRoutedExperts


def test_gradient_routed_experts_preserve_shape() -> None:
    cfg = GraftNetConfig(embed_dim=32, num_experts=4, experts_topk=2, expert_hidden_dim=64)
    module = GradientRoutedExperts(cfg)
    x = torch.randn(2, 6, 32)
    out = module(x)
    assert out.output.shape == x.shape
    assert out.routing_scores.shape == (2, 6, 4)
    assert out.selected_experts.shape == (2, 6, 2)


def test_gradient_routed_experts_ablation_uses_dense_ffn() -> None:
    cfg = GraftNetConfig(
        embed_dim=32,
        num_experts=4,
        experts_topk=2,
        expert_hidden_dim=64,
        use_gradient_routing=False,
    )
    module = GradientRoutedExperts(cfg)
    x = torch.randn(2, 6, 32)
    out = module(x)
    assert out.output.shape == x.shape
    # In ablation mode selected_experts should be zeros
    assert (out.selected_experts == 0).all()


def test_expert_load_sums_to_one() -> None:
    cfg = GraftNetConfig(embed_dim=32, num_experts=4, experts_topk=2, expert_hidden_dim=64)
    module = GradientRoutedExperts(cfg)
    x = torch.randn(3, 10, 32)
    out = module(x)
    assert torch.isclose(out.expert_load.sum(), torch.tensor(1.0), atol=1e-5)
