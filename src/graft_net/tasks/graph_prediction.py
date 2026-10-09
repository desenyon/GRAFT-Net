"""Graph Prediction task model (EXPERIMENTAL).

Treats node features as the sequence; uses the learned topology module
as the primary relational backbone.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graft_net.models.config import GraftNetConfig
from graft_net.models.heads import GraphClassificationHead
from graft_net.models.registry import build_backbone
from graft_net.tasks.auxiliary import auxiliary_outputs


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
            cfg = replace(
                cfg,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                num_classes=num_classes,
            )
        self.cfg = cfg
        self.node_proj = nn.Linear(
            cfg.node_input_dim or cfg.embed_dim, cfg.embed_dim
        )  # optional re-projection
        self.backbone = build_backbone(cfg)
        self.head = GraphClassificationHead(cfg.embed_dim, cfg.num_classes, dropout=cfg.dropout)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        batch keys:
            node_features: (B, N, D)
            adjacency:     (B, N, N) float (optional hint; topology is learned)
            labels:        (B,) long
        """
        x = batch["node_features"]  # (B, N, D)
        labels = batch["labels"]

        x_proj = self.node_proj(x)
        kwargs = (
            {"adjacency": batch.get("adjacency")}
            if self.cfg.model_name == "graph_transformer"
            else {}
        )
        backbone_out = self.backbone(x_proj, **kwargs)
        logits = self.head(backbone_out.hidden)
        task_loss = F.cross_entropy(logits, labels)

        return {
            "logits": logits,
            "task_loss": task_loss,
            **auxiliary_outputs(backbone_out, task_loss, self.cfg, self.training),
        }
