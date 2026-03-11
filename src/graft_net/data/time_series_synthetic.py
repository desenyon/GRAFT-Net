"""Synthetic time-series dataset for local smoke testing."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticTimeSeriesDataset(Dataset):
    """Autoregressive synthetic time series with sinusoidal components."""

    def __init__(
        self,
        num_samples: int = 256,
        context_len: int = 48,
        horizon: int = 12,
        input_features: int = 7,
        seed: int = 42,
    ) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        total_len = context_len + horizon
        noise = torch.randn(num_samples, total_len, input_features, generator=rng) * 0.1
        # Add sinusoidal signals with different frequencies per feature
        t = torch.arange(total_len).float()
        for f in range(input_features):
            freq = (f + 1) * 0.1
            noise[:, :, f] += torch.sin(freq * t).unsqueeze(0)
        self.inputs = noise[:, :context_len, :]
        self.targets = noise[:, context_len:, :]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {"inputs": self.inputs[idx], "targets": self.targets[idx]}
