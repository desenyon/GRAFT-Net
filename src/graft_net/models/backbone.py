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
    hidden: Tensor  # (B, N, D) final representations
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
        _b, n, d = x.shape
        if not 0 < n <= self.cfg.max_seq_len:
            raise ValueError(f"Sequence length must be in [1, {self.cfg.max_seq_len}], got {n}")
        if d != self.cfg.embed_dim:
            raise ValueError(f"Expected feature dimension {self.cfg.embed_dim}, got {d}")
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool or attention_mask.shape != x.shape[:2]:
                raise ValueError("attention_mask must be boolean with shape (batch, tokens)")
            if not attention_mask.any(-1).all():
                raise ValueError("Each sample must contain at least one unmasked token")
            x = x.masked_fill(~attention_mask.unsqueeze(-1), 0)

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
        if attention_mask is not None:
            x = x.masked_fill(~attention_mask.unsqueeze(-1), 0)
        return BackboneOutput(hidden=x, all_block_outputs=block_outputs)
