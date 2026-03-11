"""Sequence Classification task model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graft_net.models.backbone import GraftNetBackbone
from graft_net.models.config import GraftNetConfig
from graft_net.models.heads import SequenceClassificationHead


class SequenceClassificationModel(nn.Module):
    """GRAFT-Net backbone + classification head."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = 10,
        cfg: GraftNetConfig | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            from dataclasses import replace
            cfg = GraftNetConfig()
            cfg = replace(cfg, embed_dim=embed_dim, num_layers=num_layers,
                          num_heads=num_heads, num_classes=num_classes)
        self.cfg = cfg
        self.backbone = GraftNetBackbone(cfg)
        self.head = SequenceClassificationHead(cfg.embed_dim, num_classes, dropout=cfg.dropout)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        batch keys:
            inputs: (B, N, D) — pre-embedded token features
            labels: (B,) long

        Returns dict with logits, task_loss, and backbone intermediate outputs.
        """
        x = batch["inputs"]
        labels = batch["labels"]
        mask = batch.get("attention_mask", None)

        backbone_out = self.backbone(x, attention_mask=mask)
        logits = self.head(backbone_out.hidden, attention_mask=mask)
        task_loss = F.cross_entropy(logits, labels)

        # Collect outputs needed for total loss
        last_block = backbone_out.all_block_outputs[-1]
        # Use next-layer hidden as proxy future target (stop-gradiented)
        future_target = backbone_out.hidden  # for single-stage, use final hidden as target

        return {
            "logits": logits,
            "task_loss": task_loss,
            "future_state": last_block.attn_out.future_state,
            "future_target": future_target,
            "routing_scores": last_block.expert_out.routing_scores,
            "routing_targets": last_block.expert_out.routing_scores,  # placeholder; trainer computes real targets
            "soft_adjacency": last_block.topo_out.soft_adjacency,
        }
