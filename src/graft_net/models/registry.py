"""Build registered backbones from a single authoritative model configuration."""

from dataclasses import replace
from typing import Any

import torch.nn as nn

from graft_net.models.backbone import GraftNetBackbone
from graft_net.models.baselines.graph_transformer import GraphTransformer
from graft_net.models.baselines.moe_transformer import MoETransformer
from graft_net.models.baselines.transformer import StandardTransformer
from graft_net.models.config import GraftNetConfig

MODEL_NAMES = ("graft_net", "transformer", "moe_transformer", "graph_transformer")


def build_backbone(cfg: GraftNetConfig) -> nn.Module:
    name = "transformer" if cfg.model_name == "standard_transformer" else cfg.model_name
    if name == "graft_net":
        return GraftNetBackbone(cfg)
    shared: dict[str, Any] = {
        "embed_dim": cfg.embed_dim,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "dropout": cfg.dropout,
        "max_seq_len": cfg.max_seq_len,
    }
    if name == "transformer":
        return StandardTransformer(**shared, ffn_hidden=cfg.expert_hidden_dim)
    if name == "graph_transformer":
        return GraphTransformer(**shared, ffn_hidden=cfg.expert_hidden_dim)
    if name == "moe_transformer":
        return MoETransformer(
            **shared,
            num_experts=cfg.num_experts,
            experts_topk=cfg.experts_topk,
            expert_hidden=cfg.expert_hidden_dim,
        )
    raise ValueError(f"Unknown model name: {name!r}. Choose from: {', '.join(MODEL_NAMES)}")


def build_model(
    name: str, embed_dim: int = 256, num_layers: int = 4, num_heads: int = 8, **kwargs: Any
) -> nn.Module:
    """Compatibility API; legacy hidden-width aliases remain supported."""
    for alias in ("ffn_hidden", "expert_hidden"):
        if alias in kwargs:
            kwargs["expert_hidden_dim"] = kwargs.pop(alias)
    if name != "graft_net":
        kwargs.setdefault("expert_hidden_dim", embed_dim * 2)
    cfg = replace(
        GraftNetConfig(),
        model_name=name,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        **kwargs,
    )
    return build_backbone(cfg)
