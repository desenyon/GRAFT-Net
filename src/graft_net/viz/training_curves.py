"""Training curve visualisation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(
    metrics: dict[str, Sequence[float]],
    output_path: Path,
    title: str = "Training Curves",
) -> None:
    """Plot train/val loss curves.

    Args:
        metrics: Dict with keys like 'train/task_loss', 'val/task_loss' and list of values.
        output_path: Path to save the figure (.png or .pdf).
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    steps = metrics.get("step", list(range(max(len(v) for v in metrics.values() if v))))

    for key, values in metrics.items():
        if key == "step":
            continue
        ax.plot(steps[: len(values)], values, label=key)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
