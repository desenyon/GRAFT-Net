"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, step: int, output_dir: Path, tag: str = "best") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_{tag}_step{step}.pt"
    torch.save({"model_state": model.state_dict(), "step": step}, path)
    return path


def load_checkpoint(model: nn.Module, path: Path, device: torch.device | None = None) -> int:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    return ckpt.get("step", 0)
