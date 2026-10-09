"""Sample-weighted task metrics for all registered models."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None
) -> dict[str, float]:
    device = device or next(model.parameters(), torch.empty(0)).device
    was_training = model.training
    model.to(device).eval()
    total_loss = 0.0
    count = 0
    correct = 0
    absolute_error = 0.0
    elements = 0
    classification = False
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(batch)
                size = next(iter(batch.values())).shape[0]
                total_loss += outputs["task_loss"].item() * size
                count += size
                if "logits" in outputs:
                    classification = True
                    correct += (outputs["logits"].argmax(-1) == batch["labels"]).sum().item()
                if "predictions" in outputs:
                    absolute_error += (outputs["predictions"] - batch["targets"]).abs().sum().item()
                    elements += batch["targets"].numel()
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Evaluation loader is empty")
    metrics = {"eval/task_loss": total_loss / count}
    if classification:
        metrics["eval/accuracy"] = correct / count
    if elements:
        metrics["eval/mse"] = total_loss / count
        metrics["eval/mae"] = absolute_error / elements
    return metrics
