"""Regression checks for learning signals, not just output shapes."""

from dataclasses import replace

import pytest
import torch

from graft_net.losses.total import compute_total_loss
from graft_net.models.config import GraftNetConfig
from graft_net.modules.latent_topology import LatentTopologyModule
from graft_net.tasks.sequence_classification import SequenceClassificationModel


@pytest.mark.parametrize("topk", [1, 2])
def test_task_gradient_reaches_edge_scorer(topk):
    torch.manual_seed(11)
    cfg = replace(GraftNetConfig().smoke_test_variant(), topology_topk=topk, dropout=0.0)
    module = LatentTopologyModule(cfg)
    out = module(torch.randn(2, 5, cfg.embed_dim))
    out.graph_state.square().mean().backward()
    grad = module.edge_scorer.net[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    assert (out.hard_adjacency.sum(-1) == topk).all()


def test_routing_supervision_is_detached_and_trains_every_layer():
    torch.manual_seed(7)
    cfg = replace(GraftNetConfig().smoke_test_variant(), num_classes=3, dropout=0.0)
    model = SequenceClassificationModel(cfg=cfg)
    batch = {"inputs": torch.randn(2, 7, cfg.embed_dim), "labels": torch.tensor([0, 1])}
    out = model(batch)
    assert not out["routing_targets"].requires_grad
    assert not torch.allclose(out["routing_scores"], out["routing_targets"])
    loss = compute_total_loss(out)["grad_routing"]
    assert loss.item() > 1e-6
    loss.backward()
    for block in model.backbone.blocks:
        grad = block.experts.utility_predictor[-1].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0


def test_ablation_has_no_disabled_auxiliary_losses():
    cfg = replace(
        GraftNetConfig().smoke_test_variant(),
        use_predictive_attention=False,
        use_latent_topology=False,
        use_gradient_routing=False,
    )
    model = SequenceClassificationModel(cfg=cfg)
    out = model({"inputs": torch.randn(2, 5, cfg.embed_dim), "labels": torch.tensor([0, 1])})
    losses = compute_total_loss(out)
    for name in ("future", "grad_routing", "topology", "balance"):
        assert losses[name].item() == 0


def test_padding_cannot_change_predictions_or_auxiliary_losses():
    torch.manual_seed(3)
    cfg = replace(GraftNetConfig().smoke_test_variant(), dropout=0.0)
    model = SequenceClassificationModel(cfg=cfg)
    x = torch.randn(2, 6, cfg.embed_dim)
    mask = torch.tensor([[True, True, True, False, False, False]]).expand(2, -1)
    batch = {"inputs": x, "labels": torch.tensor([0, 1]), "attention_mask": mask}
    original = model(batch)
    altered = x.clone()
    altered[~mask] = 100 * torch.randn_like(altered[~mask])
    changed = model({**batch, "inputs": altered})
    torch.testing.assert_close(original["logits"], changed["logits"])
    for key, loss in compute_total_loss(original).items():
        torch.testing.assert_close(loss, compute_total_loss(changed)[key])


def test_inference_needs_no_backward_graph():
    model = SequenceClassificationModel(cfg=GraftNetConfig().smoke_test_variant()).eval()
    with torch.inference_mode():
        out = model({"inputs": torch.randn(2, 5, 32), "labels": torch.tensor([0, 1])})
    assert torch.isfinite(out["logits"]).all()
