"""Latent topology module tests."""

import torch

from graft_net.models.config import GraftNetConfig
from graft_net.modules.latent_topology import LatentTopologyModule


def test_latent_topology_returns_graph_outputs() -> None:
    cfg = GraftNetConfig(embed_dim=16, topology_topk=2, topology_edge_hidden=16)
    module = LatentTopologyModule(cfg)
    x = torch.randn(2, 5, 16)
    out = module(x)
    assert out.graph_state.shape == x.shape
    assert out.soft_adjacency.shape == (2, 5, 5)
    assert out.hard_adjacency.shape == (2, 5, 5)


def test_latent_topology_ablation_zeros_graph_state() -> None:
    cfg = GraftNetConfig(embed_dim=16, use_latent_topology=False)
    module = LatentTopologyModule(cfg)
    x = torch.randn(2, 5, 16)
    out = module(x)
    assert out.graph_state.shape == x.shape
    assert torch.all(out.graph_state == 0)


def test_latent_topology_hard_adjacency_is_bool() -> None:
    cfg = GraftNetConfig(embed_dim=16, topology_topk=2, topology_edge_hidden=16)
    module = LatentTopologyModule(cfg)
    x = torch.randn(1, 6, 16)
    out = module(x)
    assert out.hard_adjacency.dtype == torch.bool


def test_half_precision_masked_topology_diagnostics_are_finite():
    cfg = GraftNetConfig(embed_dim=16, topology_topk=2, topology_edge_hidden=16)
    module = LatentTopologyModule(cfg).half()
    mask = torch.tensor([[True, True, False, False]])
    out = module(torch.randn(1, 4, 16).half(), attention_mask=mask)
    assert torch.isfinite(out.graph_entropy)
    assert torch.isfinite(out.graph_state).all()
