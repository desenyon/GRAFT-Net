"""Model registry: build any model by name."""

from __future__ import annotations

import torch.nn as nn

from graft_net.models.baselines.graph_transformer import GraphTransformer
from graft_net.models.baselines.moe_transformer import MoETransformer
from graft_net.models.baselines.transformer import StandardTransformer
from graft_net.models.backbone import GraftNetBackbone
from graft_net.models.config import GraftNetConfig


def build_model(
    name: str,
    embed_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    **kwargs: object,
) -> nn.Module:
    """Build a model by name.

    Supported names: graft_net, transformer, moe_transformer, graph_transformer.
    """
    if name == "graft_net":
        from dataclasses import replace
        cfg = GraftNetConfig()
        cfg = replace(cfg, embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads)
        for key, val in kwargs.items():
            if hasattr(cfg, key):
                cfg = replace(cfg, **{key: val})
        return GraftNetBackbone(cfg)

    if name == "transformer":
        return StandardTransformer(
            embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads,
            ffn_hidden=int(kwargs.get("ffn_hidden", embed_dim * 2)),  # type: ignore[arg-type]
        )

    if name == "moe_transformer":
        return MoETransformer(
            embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads,
            num_experts=int(kwargs.get("num_experts", 8)),  # type: ignore[arg-type]
            experts_topk=int(kwargs.get("experts_topk", 2)),  # type: ignore[arg-type]
            expert_hidden=int(kwargs.get("expert_hidden", embed_dim * 2)),  # type: ignore[arg-type]
        )

    if name == "graph_transformer":
        return GraphTransformer(
            embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads,
            ffn_hidden=int(kwargs.get("ffn_hidden", embed_dim * 2)),  # type: ignore[arg-type]
        )

    raise ValueError(f"Unknown model name: {name!r}. "
                     f"Choose from: graft_net, transformer, moe_transformer, graph_transformer")
