"""Latent Topology Learning module.

Learns a sparse directed graph over tokens and performs message passing.
Graph contribution is fused with attention output via a gated residual.
Ablation path (use_latent_topology=False): graph_state=zeros, adjacency=identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from graft_net.models.config import GraftNetConfig
from graft_net.topology.edge_scorer import MLPEdgeScorer
from graft_net.topology.message_passing import GraphMessagePassing
from graft_net.topology.sparsify import topk_adjacency
from graft_net.utils.tensor import masked_softmax


@dataclass
class TopologyOutput:
    graph_state: Tensor  # (B, N, D) post-aggregation node representations
    soft_adjacency: Tensor  # (B, N, N) softmax-normalised edge weights
    hard_adjacency: Tensor  # (B, N, N) bool top-k adjacency
    graph_entropy: Tensor  # () scalar, mean per-node edge distribution entropy
    mean_degree: Tensor  # () scalar, mean out-degree


class LatentTopologyModule(nn.Module):
    """Latent topology learning: edge scoring → sparsification → message passing."""

    def __init__(self, cfg: GraftNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim

        self.edge_scorer = MLPEdgeScorer(d, cfg.topology_edge_hidden)
        self.message_passing = GraphMessagePassing(d)
        # Gating: learned scalar gate per token
        self.gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> TopologyOutput:
        b, n, _d = x.shape

        if not self.cfg.use_latent_topology:
            zeros = torch.zeros_like(x)
            identity = (
                torch.eye(n, device=x.device, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
            )
            identity_float = identity.float()
            return TopologyOutput(
                graph_state=zeros,
                soft_adjacency=identity_float,
                hard_adjacency=identity,
                graph_entropy=torch.tensor(0.0, device=x.device),
                mean_degree=torch.tensor(1.0, device=x.device),
            )

        # Score edges
        edge_logits = self.edge_scorer(x)  # (B, N, N)
        valid = torch.ones_like(edge_logits, dtype=torch.bool)
        if attention_mask is not None:
            valid = attention_mask[:, :, None] & attention_mask[:, None, :]
        soft_adj = masked_softmax(edge_logits, valid)
        hard_adj = (
            topk_adjacency(edge_logits.masked_fill(~valid, float("-inf")), k=self.cfg.topology_topk)
            & valid
        )

        # Diagnostics
        eps = 1e-8
        entropy = -(soft_adj.float() * (soft_adj.float() + eps).log()).sum(-1).mean()  # entropy
        mean_deg = hard_adj.float().sum(-1).mean()

        # Hard uniform top-k forward, soft surrogate backward. This keeps a task
        # gradient to the scorer even at k=1; it is a biased straight-through estimator.
        hard_weights = hard_adj.to(x.dtype)
        hard_weights = hard_weights / hard_weights.sum(-1, keepdim=True).clamp(min=1)
        adjacency = hard_weights + (soft_adj - soft_adj.detach())
        graph_state = self.message_passing(x, adjacency)

        # Gated fusion: model decides how much topology dominates
        gate_input = torch.cat([x, graph_state], dim=-1)
        gate = self.gate(gate_input)  # (B, N, D)
        graph_state = gate * graph_state
        if attention_mask is not None:
            graph_state = graph_state * attention_mask.unsqueeze(-1)

        return TopologyOutput(
            graph_state=graph_state,
            soft_adjacency=soft_adj,
            hard_adjacency=hard_adj,
            graph_entropy=entropy,
            mean_degree=mean_deg,
        )
