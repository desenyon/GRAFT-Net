"""Model config tests."""

from graft_net.models.config import GraftNetConfig


def test_model_config_has_expected_defaults() -> None:
    cfg = GraftNetConfig()
    assert cfg.embed_dim > 0
    assert cfg.num_layers > 0
    assert cfg.num_experts > 0
    assert cfg.use_predictive_attention is True
    assert cfg.use_latent_topology is True
    assert cfg.use_gradient_routing is True


def test_smoke_test_variant_is_smaller() -> None:
    cfg = GraftNetConfig()
    small = cfg.smoke_test_variant()
    assert small.embed_dim < cfg.embed_dim
    assert small.num_layers <= cfg.num_layers
