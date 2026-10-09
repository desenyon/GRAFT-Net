"""Exercise resolved Hydra configs through real construction and checkpoints."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from graft_net.train.trainer import Trainer

CONFIGS = str(Path(__file__).resolve().parents[2] / "configs")


def compose_test_config(tmp_path, overrides=()):
    with initialize_config_dir(config_dir=CONFIGS, version_base="1.3"):
        return compose("config", overrides=["compute=smoke", f"output_dir={tmp_path}", *overrides])


@pytest.mark.parametrize(
    "task", ["sequence_classification", "time_series_forecasting", "graph_prediction"]
)
@pytest.mark.parametrize(
    "name,backbone",
    [
        ("graft_net", "GraftNetBackbone"),
        ("transformer", "StandardTransformer"),
        ("moe_transformer", "MoETransformer"),
        ("graph_transformer", "GraphTransformer"),
    ],
)
def test_each_task_and_model_trains_the_selected_architecture(tmp_path, task, name, backbone):
    cfg = compose_test_config(
        tmp_path,
        [f"task={task}", f"model.model_name={name}", "model.embed_dim=16", "model.num_heads=2"],
    )
    trainer = Trainer.from_config(cfg)
    assert trainer.cfg.embed_dim == 16
    assert type(trainer.model.backbone).__name__ == backbone
    history = trainer.train(1)
    assert history["train/total_loss"][0] >= 0
    assert history["val/task_loss"][0] >= 0
    checkpoint = trainer.save_checkpoint()
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["model_config"]["model_name"] == name
    restored = Trainer.from_checkpoint(checkpoint, output_dir=tmp_path / "restored")
    assert restored.evaluate() == trainer.evaluate()
    assert restored._step == trainer._step
    assert restored.optimizer.state_dict()["state"]


def test_compute_defaults_and_explicit_model_overrides(tmp_path):
    cfg = compose_test_config(
        tmp_path, ["model.embed_dim=48", "model.num_heads=4", "compute.learning_rate=0.003"]
    )
    trainer = Trainer.from_config(cfg)
    assert trainer.cfg.embed_dim == 48
    assert trainer.cfg.learning_rate == 0.003
    assert trainer.optimizer.param_groups[0]["lr"] == 0.003


def test_datasets_and_shuffle_are_independent_of_model_initialization(tmp_path):
    a = Trainer.from_config(compose_test_config(tmp_path / "a", ["model.model_name=graft_net"]))
    b = Trainer.from_config(compose_test_config(tmp_path / "b", ["model.model_name=transformer"]))
    batch_a, batch_b = next(iter(a.train_loader)), next(iter(b.train_loader))
    for key, value in batch_a.items():
        torch.testing.assert_close(value, batch_b[key])


@pytest.mark.parametrize(
    "override,match",
    [
        ("model.embed_dim=31", "divisible"),
        ("model.experts_topk=99", "experts_topk"),
        ("task.dataset=unknown", "synthetic"),
        ("model.max_seq_len=2", "max_seq_len"),
    ],
)
def test_invalid_config_fails_early(tmp_path, override, match):
    with pytest.raises(ValueError, match=match):
        Trainer.from_config(compose_test_config(tmp_path, [override]))


@pytest.mark.parametrize(
    "variant,flag",
    [
        ("no_predictive_attention", "use_predictive_attention"),
        ("no_latent_topology", "use_latent_topology"),
        ("no_gradient_routing", "use_gradient_routing"),
    ],
)
def test_ablation_group_reaches_constructed_modules(tmp_path, variant, flag):
    trainer = Trainer.from_config(compose_test_config(tmp_path, [f"+ablation={variant}"]))
    assert getattr(trainer.cfg, flag) is False
    assert getattr(trainer.model.backbone.blocks[0].attention.cfg, flag) is False


def test_run_evidence_does_not_require_git_executable(tmp_path, monkeypatch):
    from graft_net.experiments import run_experiment

    monkeypatch.setenv("PATH", "")
    config = compose_test_config(tmp_path)
    result = run_experiment(config, 1, tmp_path)
    assert result["environment"]["git_commit"] is None
    assert result["environment"]["git_dirty"] is None
    assert (tmp_path / "metrics.json").exists()
