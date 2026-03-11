"""Graph Prediction task model (EXPERIMENTAL).

Treats node features as the sequence; uses the learned topology module
as the primary relational backbone.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graft_net.models.backbone import GraftNetBackbone
from graft_net.models.config import GraftNetConfig
from graft_net.models.heads import GraphClassificationHead


class GraphPredictionModel(nn.Module):
    """GRAFT-Net backbone + graph-level classification head.

    Status: EXPERIMENTAL — results may lag sequence and time-series tracks.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = 2,
        cfg: GraftNetConfig | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            from dataclasses import replace
            cfg = GraftNetConfig()
            cfg = replace(cfg, embed_dim=embed_dim, num_layers=num_layers,
                          num_heads=num_heads, num_classes=num_classes)
        self.cfg = cfg
        self.node_proj = nn.Linear(embed_dim, embed_dim)  # optional re-projection
        self.backbone = GraftNetBackbone(cfg)
        self.head = GraphClassificationHead(embed_dim, num_classes, dropout=cfg.dropout)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        batch keys:
            node_features: (B, N, D)
            adjacency:     (B, N, N) float (optional hint; topology is learned)
            labels:        (B,) long
        """
        x = batch["node_features"]               # (B, N, D)
        labels = batch["labels"]

        x_proj = self.node_proj(x)
        backbone_out = self.backbone(x_proj)
        logits = self.head(backbone_out.hidden)
        task_loss = F.cross_entropy(logits, labels)

        last_block = backbone_out.all_block_outputs[-1]
        return {
            "logits": logits,
            "task_loss": task_loss,
            "future_state": last_block.attn_out.future_state,
            "future_target": backbone_out.hidden,
            "routing_scores": last_block.expert_out.routing_scores,
            "routing_targets": last_block.expert_out.routing_scores,
            "soft_adjacency": last_block.topo_out.soft_adjacency,
        }
