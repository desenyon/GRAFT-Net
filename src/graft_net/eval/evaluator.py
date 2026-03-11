"""Evaluation utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run evaluation loop and return aggregated metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    total_task_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs: dict[str, Any] = model(batch)
            total_task_loss += outputs["task_loss"].item()
            n_batches += 1

    return {"eval/task_loss": total_task_loss / max(n_batches, 1)}
