"""Gradient-Routed Expert layer.

Experts are selected by predicted gradient utility (H2 hypothesis).
Ablation path (use_gradient_routing=False): falls back to a single dense FFN.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from graft_net.models.config import GraftNetConfig
from graft_net.routing.gradient_router import topk_route


@dataclass
class ExpertOutput:
    output: Tensor  # (B, N, D)
    routing_scores: Tensor  # (B, N, E) predicted utility logits
    selected_experts: Tensor  # (B, N, k) indices of chosen experts
    expert_load: Tensor  # (E,) fraction of tokens routed to each expert
    expert_values: Tensor | None = None  # detached candidate outputs (B, N, E, D)


class _Expert(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GradientRoutedExperts(nn.Module):
    """MoE layer with gradient-utility-based routing."""

    def __init__(self, cfg: GraftNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        e = cfg.num_experts

        # Utility predictor: predicts learned gradient usefulness per expert
        self.utility_predictor = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, e),
        )

        # Expert pool
        self.experts = nn.ModuleList(
            [_Expert(d, cfg.expert_hidden_dim, d, cfg.dropout) for _ in range(e)]
        )

        # Dense FFN fallback (used when use_gradient_routing=False)
        self.dense_ffn = nn.Sequential(
            nn.Linear(d, cfg.expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.expert_hidden_dim, d),
        )

    def forward(self, x: Tensor) -> ExpertOutput:
        b, n, _d = x.shape
        e = self.cfg.num_experts
        k = self.cfg.experts_topk

        routing_scores = self.utility_predictor(x)  # (B, N, E)

        if not self.cfg.use_gradient_routing:
            # Dense FFN bypass — routing scores still computed for loss compatibility
            output = self.dense_ffn(x)
            return ExpertOutput(
                output=output,
                routing_scores=routing_scores,
                selected_experts=torch.zeros(b, n, k, dtype=torch.long, device=x.device),
                expert_load=torch.full((e,), 1.0 / e, device=x.device),
            )

        weights, indices = topk_route(routing_scores, k=k)  # (B,N,k), (B,N,k)

        # Dense candidate evaluation provides counterfactual values for utility
        # supervision. Dispatch is top-k, but this reference implementation is
        # deliberately not a sparse-compute MoE kernel.
        values = torch.stack([expert(x) for expert in self.experts], dim=-2)
        dense_weights = torch.zeros_like(routing_scores).scatter(-1, indices, weights)
        output = (dense_weights.unsqueeze(-1) * values).sum(-2)
        expert_counts = torch.bincount(indices.flatten(), minlength=e).to(x.dtype)

        total = expert_counts.sum().clamp(min=1.0)
        expert_load = expert_counts / total

        return ExpertOutput(
            output=output,
            routing_scores=routing_scores,
            selected_experts=indices,
            expert_load=expert_load,
            expert_values=values.detach(),
        )
