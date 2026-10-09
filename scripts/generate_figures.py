"""Plot recorded history, or explicitly request labeled illustrative diagrams."""

import argparse
import json
from pathlib import Path

import torch

from graft_net.viz.gradient_utility import plot_gradient_utility
from graft_net.viz.routing_patterns import plot_routing_heatmap
from graft_net.viz.topology_graphs import plot_topology_graph
from graft_net.viz.training_curves import plot_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--history", type=Path, help="history.json produced by Trainer")
    source.add_argument("--demo", action="store_true", help="Generate labeled artificial examples")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.history:
        metrics = json.loads(args.history.read_text())
        train, val = metrics["train/task_loss"], metrics["val/task_loss"]
        if not train or len(train) != len(val):
            parser.error(
                "history must contain equally sized nonempty train/task_loss and val/task_loss"
            )
        metrics["step"] = list(range(1, len(train) + 1))
        plot_training_curves(
            metrics, output / "training_curves.png", title="Recorded task losses (epochs)"
        )
        return
    torch.manual_seed(42)
    steps = list(range(20))
    metrics = {
        "step": steps,
        "train/task_loss": [1.5 * 0.9**i for i in steps],
        "val/task_loss": [1.6 * 0.91**i for i in steps],
    }
    plot_training_curves(
        metrics, output / "demo_training_curves.png", title="DEMO: artificial curves"
    )
    plot_topology_graph(
        torch.rand(1, 8, 8) > 0.7, output / "demo_topology_graph.png", title="DEMO: random topology"
    )
    plot_routing_heatmap(
        torch.randn(1, 16, 8), output / "demo_routing_heatmap.png", title="DEMO: random routing"
    )
    predicted, actual = torch.randn(32), torch.randn(32)
    plot_gradient_utility(
        predicted, actual, output / "demo_gradient_utility.png", title="DEMO: random utility values"
    )


if __name__ == "__main__":
    main()
