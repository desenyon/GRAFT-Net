"""Predictive attention module tests."""

import torch

from graft_net.models.config import GraftNetConfig
from graft_net.modules.predictive_attention import PredictiveAttention


def test_predictive_attention_returns_expected_shapes() -> None:
    cfg = GraftNetConfig(embed_dim=32, num_heads=4, predictor_hidden_dim=64)
    module = PredictiveAttention(cfg)
    x = torch.randn(2, 8, 32)
    out = module(x)
    assert out.attended.shape == x.shape
    assert out.future_state.shape == x.shape
    assert out.attention_weights.shape == (2, 4, 8, 8)


def test_predictive_attention_can_bypass_prediction() -> None:
    cfg = GraftNetConfig(embed_dim=32, num_heads=4, use_predictive_attention=False)
    module = PredictiveAttention(cfg)
    x = torch.randn(2, 8, 32)
    out = module(x)
    assert out.attended.shape == x.shape
    # In bypass mode, future_state should equal x
    assert torch.allclose(out.future_state, x)


def test_predictive_attention_with_mask() -> None:
    cfg = GraftNetConfig(embed_dim=32, num_heads=4)
    module = PredictiveAttention(cfg)
    x = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[0, 6:] = False  # mask last 2 tokens for first sample
    out = module(x, attention_mask=mask)
    assert out.attended.shape == x.shape
