"""Backbone tests."""

import torch

from graft_net.models.backbone import GraftNetBackbone
from graft_net.models.config import GraftNetConfig


def _small_cfg(**kwargs: object) -> GraftNetConfig:
    from dataclasses import replace
    base = GraftNetConfig()
    return replace(base.smoke_test_variant(), **kwargs)


def test_backbone_forward_shape() -> None:
    cfg = _small_cfg()
    model = GraftNetBackbone(cfg)
    x = torch.randn(2, 8, cfg.embed_dim)
    out = model(x)
    assert out.hidden.shape == (2, 8, cfg.embed_dim)
    assert len(out.all_block_outputs) == cfg.num_layers


def test_backbone_supports_module_drop_ablation() -> None:
    cfg = _small_cfg(
        use_predictive_attention=False,
        use_latent_topology=False,
        use_gradient_routing=False,
    )
    model = GraftNetBackbone(cfg)
    x = torch.randn(2, 8, cfg.embed_dim)
    out = model(x)
    assert out.hidden.shape == (2, 8, cfg.embed_dim)
