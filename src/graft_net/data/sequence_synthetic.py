"""Synthetic sequence classification dataset for local smoke testing."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticSequenceDataset(Dataset):
    """Random sequences with class-correlated patterns."""

    def __init__(
        self,
        num_samples: int = 256,
        seq_len: int = 32,
        embed_dim: int = 32,
        num_classes: int = 4,
        seed: int = 42,
    ) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=rng)
        # Class-correlated pattern: shift mean per class
        self.inputs = torch.randn(num_samples, seq_len, embed_dim, generator=rng)
        class_offsets = torch.randn(num_classes, embed_dim, generator=rng) * 0.5
        self.inputs += class_offsets[self.labels].unsqueeze(1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {"inputs": self.inputs[idx], "labels": self.labels[idx]}
