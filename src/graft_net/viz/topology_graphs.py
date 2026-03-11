"""Learned topology graph visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor


def plot_topology_graph(
    adjacency: Tensor,
    output_path: Path,
    title: str = "Learned Latent Topology",
    sample_idx: int = 0,
) -> None:
    """Render learned hard adjacency as a directed graph.

    Args:
        adjacency: (B, N, N) boolean or float adjacency tensor.
        output_path: Where to save the figure.
        sample_idx: Which batch element to visualise.
        title: Plot title.
    """
    adj = adjacency[sample_idx].detach().cpu().float().numpy()
    n = adj.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0.5:
                G.add_edge(i, j, weight=float(adj[i, j]))

    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos=pos, ax=ax, node_size=200, arrows=True,
                     node_color="#4C72B0", edge_color="#999999", font_size=8)
    ax.set_title(title)
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
