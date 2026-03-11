"""Generate and save all paper figures to ``outputs/figures/``.

Usage::

    python scripts/generate_figures.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from graft_net.viz.training_curves import plot_training_curves
    from graft_net.viz.topology_graphs import plot_topology_graph
    from graft_net.viz.routing_patterns import plot_routing_heatmap
    from graft_net.viz.gradient_utility import plot_gradient_utility

    out = Path("outputs/figures")
    out.mkdir(parents=True, exist_ok=True)

    # ---- training curves (demo data) ----
    steps = list(range(20))
    metrics = {
        "step": steps,
        "train/task_loss": [1.5 * (0.9 ** i) for i in steps],
        "val/task_loss": [1.6 * (0.91 ** i) for i in steps],
    }
    plot_training_curves(metrics, out / "training_curves.png", title="GRAFT-Net Training")
    log.info("Saved training_curves.png")

    # ---- topology graph ----
    adj = (torch.rand(1, 8, 8) > 0.7)
    plot_topology_graph(adj, out / "topology_graph.png")
    log.info("Saved topology_graph.png")

    # ---- routing heatmap ----
    routing = torch.softmax(torch.randn(1, 16, 8), dim=-1)
    plot_routing_heatmap(routing, out / "routing_heatmap.png")
    log.info("Saved routing_heatmap.png")

    # ---- gradient utility scatter ----
    predicted = torch.randn(32)
    actual = predicted + 0.1 * torch.randn(32)
    plot_gradient_utility(predicted, actual, out / "gradient_utility.png")
    log.info("Saved gradient_utility.png")

    log.info("All figures written to %s", out.resolve())


if __name__ == "__main__":
    main()
