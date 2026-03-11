"""Task-specific classification and forecasting heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SequenceClassificationHead(nn.Module):
    """Mean-pool backbone output → linear classifier."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            hidden: (B, N, D)
            attention_mask: (B, N) bool, True = real token
        Returns:
            logits: (B, num_classes)
        """
        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)
        return self.classifier(self.dropout(pooled))


class ForecastingHead(nn.Module):
    """Linear projection from last backbone features to multi-step forecast."""

    def __init__(self, embed_dim: int, input_features: int, horizon: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, input_features * horizon)
        self.horizon = horizon
        self.input_features = input_features

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Args:
            hidden: (B, N, D)
        Returns:
            predictions: (B, horizon, input_features)
        """
        pooled = hidden[:, -1, :]                          # use last token
        out = self.proj(pooled)                            # (B, horizon * features)
        return out.view(hidden.shape[0], self.horizon, self.input_features)


class GraphClassificationHead(nn.Module):
    """Mean-pool over all nodes → linear classifier for graph-level tasks."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Args:
            hidden: (B, N, D)
        Returns:
            logits: (B, num_classes)
        """
        pooled = hidden.mean(dim=1)
        return self.classifier(self.dropout(pooled))
