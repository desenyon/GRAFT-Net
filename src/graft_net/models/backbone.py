"""GRAFT-Net Backbone: stack of GRAFT blocks with positional embedding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from graft_net.models.block import BlockOutput, GraftBlock
from graft_net.models.config import GraftNetConfig


@dataclass
class BackboneOutput:
    hidden: Tensor                      # (B, N, D) final representations
    all_block_outputs: list[BlockOutput]  # one per layer, for diagnostics


class GraftNetBackbone(nn.Module):
    """Full GRAFT-Net backbone: embedding projection + N GRAFT blocks + final norm."""

    def __init__(self, cfg: GraftNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim

        self.pos_embedding = nn.Embedding(cfg.max_seq_len, d)
        self.input_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([GraftBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(d)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> BackboneOutput:
        b, n, d = x.shape

        # Add positional embeddings
        positions = torch.arange(n, device=x.device).unsqueeze(0)  # (1, N)
        x = x + self.pos_embedding(positions)
        x = self.input_dropout(x)

        block_outputs: list[BlockOutput] = []
        for block in self.blocks:
            bout = block(x, attention_mask=attention_mask)
            x = bout.hidden
            block_outputs.append(bout)

        x = self.final_norm(x)
        return BackboneOutput(hidden=x, all_block_outputs=block_outputs)
