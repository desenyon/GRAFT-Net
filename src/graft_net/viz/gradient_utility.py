"""Gradient utility visualisation: predicted vs actual utility scatter."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_gradient_utility(
    predicted: Tensor,
    actual: Tensor,
    output_path: Path,
    title: str = "Predicted vs Actual Gradient Utility",
) -> None:
    """Scatter plot comparing predicted utility scores to gradient-derived targets.

    Args:
        predicted: (B, N, E) utility logits from predictor.
        actual: (B, N, E) reference targets (gradient magnitudes or proxies).
        output_path: Where to save.
        title: Plot title.
    """
    import torch.nn.functional as F

    pred_flat = F.softmax(predicted.detach().cpu().reshape(-1, predicted.shape[-1]), dim=-1).numpy().ravel()
    actual_flat = F.softmax(actual.detach().cpu().reshape(-1, actual.shape[-1]), dim=-1).numpy().ravel()

    # Sub-sample for visibility
    max_pts = 5000
    if len(pred_flat) > max_pts:
        idx = torch.randperm(len(pred_flat))[:max_pts].numpy()
        pred_flat = pred_flat[idx]
        actual_flat = actual_flat[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual_flat, pred_flat, alpha=0.3, s=5)
    lim = [min(actual_flat.min(), pred_flat.min()), max(actual_flat.max(), pred_flat.max())]
    ax.plot(lim, lim, "r--", linewidth=1, label="y=x (perfect)")
    ax.set_xlabel("Actual utility (normalised)")
    ax.set_ylabel("Predicted utility (normalised)")
    ax.set_title(title)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
