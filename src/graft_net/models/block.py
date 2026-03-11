"""GRAFT-Net Block: assembles predictive attention + topology fusion + expert layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from graft_net.models.config import GraftNetConfig
from graft_net.modules.experts import ExpertOutput, GradientRoutedExperts
from graft_net.modules.latent_topology import LatentTopologyModule, TopologyOutput
from graft_net.modules.predictive_attention import PredictiveAttention, PredictiveAttentionOutput


@dataclass
class BlockOutput:
    hidden: Tensor                          # (B, N, D) output
    attn_out: PredictiveAttentionOutput
    topo_out: TopologyOutput
    expert_out: ExpertOutput


class GraftBlock(nn.Module):
    """One GRAFT-Net block following the design sequence:
    normalize → predict future → attend → topology → fuse → experts → residual.
    """

    def __init__(self, cfg: GraftNetConfig) -> None:
        super().__init__()
        d = cfg.embed_dim

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

        self.attention = PredictiveAttention(cfg)
        self.topology = LatentTopologyModule(cfg)

        # Fusion gate: merges attention output and topology output
        self.fusion_gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.Sigmoid(),
        )

        self.experts = GradientRoutedExperts(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> BlockOutput:
        # 1. Pre-norm + predictive attention + residual
        normed = self.norm1(x)
        attn_out = self.attention(normed, attention_mask=attention_mask)
        x = x + self.dropout(attn_out.attended)

        # 2. Pre-norm + latent topology + residual
        normed = self.norm2(x)
        topo_out = self.topology(normed)

        # 3. Gated fusion of attention and topology streams
        gate = self.fusion_gate(torch.cat([attn_out.attended, topo_out.graph_state], dim=-1))
        fused = attn_out.attended + gate * topo_out.graph_state
        x = x + self.dropout(fused - attn_out.attended)  # add delta cleanly

        # 4. Pre-norm + experts + residual
        normed = self.norm3(x)
        expert_out = self.experts(normed)
        x = x + self.dropout(expert_out.output)

        return BlockOutput(hidden=x, attn_out=attn_out, topo_out=topo_out, expert_out=expert_out)
