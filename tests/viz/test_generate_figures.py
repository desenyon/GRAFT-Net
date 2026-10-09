"""Visualization tests."""

from pathlib import Path

import torch

from graft_net.viz.training_curves import plot_training_curves


def test_plot_training_curves_writes_output(tmp_path: Path) -> None:
    metrics = {
        "step": [0, 1, 2],
        "train/task_loss": [1.0, 0.8, 0.6],
        "val/task_loss": [1.1, 0.9, 0.7],
    }
    output_path = tmp_path / "curve.png"
    plot_training_curves(metrics, output_path)
    assert output_path.exists()


def test_plot_topology_graph_writes_output(tmp_path: Path) -> None:
    from graft_net.viz.topology_graphs import plot_topology_graph

    adj = torch.rand(2, 5, 5) > 0.6
    output_path = tmp_path / "topology.png"
    plot_topology_graph(adj, output_path)
    assert output_path.exists()


def test_plot_routing_heatmap_writes_output(tmp_path: Path) -> None:
    from graft_net.viz.routing_patterns import plot_routing_heatmap

    scores = torch.randn(2, 12, 8)
    output_path = tmp_path / "routing.png"
    plot_routing_heatmap(scores, output_path)
    assert output_path.exists()
