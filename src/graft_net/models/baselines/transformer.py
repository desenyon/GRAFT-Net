"""Standard Transformer baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TransformerOutput:
    hidden: Tensor


class StandardTransformer(nn.Module):
    """Vanilla Transformer without any GRAFT-Net novelty."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_hidden: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> TransformerOutput:
        _b, n, _d = x.shape
        positions = torch.arange(n, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        key_padding_mask = ~attention_mask if attention_mask is not None else None
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return TransformerOutput(hidden=self.norm(out))
