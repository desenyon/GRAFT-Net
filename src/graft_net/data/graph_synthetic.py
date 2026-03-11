"""Synthetic graph classification dataset for local smoke testing (EXPERIMENTAL)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticGraphDataset(Dataset):
    """Random graphs with class-correlated node feature statistics."""

    def __init__(
        self,
        num_samples: int = 256,
        num_nodes: int = 16,
        embed_dim: int = 32,
        num_classes: int = 2,
        edge_prob: float = 0.3,
        seed: int = 42,
    ) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=rng)
        self.node_features = torch.randn(num_samples, num_nodes, embed_dim, generator=rng)
        class_offsets = torch.randn(num_classes, embed_dim, generator=rng) * 0.5
        self.node_features += class_offsets[self.labels].unsqueeze(1)
        # Random adjacency
        adj = torch.rand(num_samples, num_nodes, num_nodes, generator=rng) < edge_prob
        self.adjacency = (adj | adj.transpose(-1, -2)).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "node_features": self.node_features[idx],
            "adjacency": self.adjacency[idx],
            "labels": self.labels[idx],
        }
