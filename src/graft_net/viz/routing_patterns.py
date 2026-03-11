"""Expert routing pattern visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_routing_heatmap(
    routing_scores: Tensor,
    output_path: Path,
    title: str = "Expert Routing Heatmap",
    sample_idx: int = 0,
) -> None:
    """Plot token × expert routing probability heatmap.

    Args:
        routing_scores: (B, N, E) routing logits or probabilities.
        output_path: Where to save the figure.
        sample_idx: Batch element to visualise.
        title: Plot title.
    """
    import torch.nn.functional as F

    probs = F.softmax(routing_scores[sample_idx].detach().cpu(), dim=-1).numpy()  # (N, E)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(probs.T, aspect="auto", cmap="viridis")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Expert index")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Routing probability")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
