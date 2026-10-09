"""Metric weighting, deterministic resume, and error handling regressions."""

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from graft_net.eval.evaluator import evaluate
from graft_net.losses.gradient_prediction import gradient_prediction_loss
from graft_net.losses.topology import topology_loss
from graft_net.models.config import GraftNetConfig
from graft_net.models.registry import build_backbone
from graft_net.tasks.sequence_classification import SequenceClassificationModel
from graft_net.train.trainer import Trainer


def test_resume_matches_uninterrupted_cpu_training(tmp_path):
    uninterrupted = Trainer.for_smoke_test(tmp_path / "full")
    uninterrupted.train(2)
    interrupted = Trainer.for_smoke_test(tmp_path / "split")
    interrupted.train(1)
    restored = Trainer.from_checkpoint(
        interrupted.save_checkpoint(), output_dir=tmp_path / "resume"
    )
    restored.train(1)
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=0)
    assert restored.history["val/task_loss"] == uninterrupted.history["val/task_loss"]


class ScalarLoss(nn.Module):
    def forward(self, batch):
        return {"task_loss": batch["value"].mean()}


def test_evaluation_weights_last_short_batch_and_restores_mode():
    model = ScalarLoss().train()
    loader = DataLoader([{"value": torch.tensor(float(i))} for i in range(5)], batch_size=2)
    assert evaluate(model, loader)["eval/task_loss"] == 2.0
    assert model.training
    with pytest.raises(ValueError, match="empty"):
        evaluate(model, DataLoader([]))
    assert model.training


def test_router_kl_is_independent_of_repeated_tokens():
    scores, targets = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    torch.testing.assert_close(
        gradient_prediction_loss(scores, targets),
        gradient_prediction_loss(scores.repeat(1, 5, 1), targets.repeat(1, 5, 1)),
    )


def test_topology_regularizer_measures_entropy_not_constant_l1():
    peaked = torch.eye(4).unsqueeze(0)
    uniform = torch.full((1, 4, 4), 0.25)
    assert topology_loss(peaked).item() == 0
    assert topology_loss(uniform).item() > 1


@pytest.mark.parametrize(
    "name", ["graft_net", "transformer", "moe_transformer", "graph_transformer"]
)
def test_backbone_padding_invariance(name):
    cfg = replace(GraftNetConfig().smoke_test_variant(), model_name=name, dropout=0.0)
    model = build_backbone(cfg).eval()
    x = torch.randn(2, 5, 32)
    mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)
    changed = x.clone()
    changed[~mask] = 100
    with torch.no_grad():
        a, b = model(x, attention_mask=mask).hidden, model(changed, attention_mask=mask).hidden
    torch.testing.assert_close(a[mask], b[mask])


def test_router_target_responds_to_task_labels():
    torch.manual_seed(13)
    cfg = replace(GraftNetConfig().smoke_test_variant(), dropout=0.0, num_classes=3)
    model = SequenceClassificationModel(cfg=cfg)
    x = torch.randn(2, 5, 32)
    a = model({"inputs": x, "labels": torch.tensor([0, 0])})
    b = model({"inputs": x, "labels": torch.tensor([1, 2])})
    torch.testing.assert_close(a["routing_scores"], b["routing_scores"])
    assert not torch.allclose(a["routing_targets"], b["routing_targets"])


def test_checkpoint_rejects_mismatched_configuration(tmp_path):
    a = Trainer.for_smoke_test(tmp_path / "a", num_classes=3)
    b = Trainer.for_smoke_test(tmp_path / "b", num_classes=4)
    with pytest.raises(ValueError, match="config differs"):
        b.load_checkpoint(a.save_checkpoint())
