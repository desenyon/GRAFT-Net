"""The selected task and full supplied config must reach the actual model."""

from dataclasses import replace

import pytest
import torch

from graft_net.models.config import GraftNetConfig
from graft_net.tasks.graph_prediction import GraphPredictionModel
from graft_net.tasks.sequence_classification import SequenceClassificationModel
from graft_net.tasks.time_series_forecasting import TimeSeriesForecastingModel
from graft_net.train.trainer import Trainer


@pytest.mark.parametrize(
    "task,model_type,key",
    [
        ("sequence_classification", SequenceClassificationModel, "labels"),
        ("time_series_forecasting", TimeSeriesForecastingModel, "targets"),
        ("graph_prediction", GraphPredictionModel, "node_features"),
    ],
)
def test_smoke_factory_dispatches_real_task(tmp_path, task, model_type, key):
    trainer = Trainer.for_smoke_test(tmp_path, task_name=task)
    assert isinstance(trainer.model, model_type)
    assert key in next(iter(trainer.train_loader))
    assert trainer.run_smoke_epoch()["val/task_loss"] >= 0


def test_explicit_config_controls_task_head_dimensions():
    cfg = replace(
        GraftNetConfig().smoke_test_variant(), num_classes=3, input_features=2, forecast_horizon=4
    )
    seq = SequenceClassificationModel(cfg=cfg)
    graph = GraphPredictionModel(cfg=cfg)
    ts = TimeSeriesForecastingModel(cfg=cfg)
    labels = torch.tensor([0, 2])
    assert seq({"inputs": torch.randn(2, 5, 32), "labels": labels})["logits"].shape == (2, 3)
    assert graph({"node_features": torch.randn(2, 5, 32), "labels": labels})["logits"].shape == (
        2,
        3,
    )
    assert ts({"inputs": torch.randn(2, 5, 2), "targets": torch.randn(2, 4, 2)})[
        "predictions"
    ].shape == (2, 4, 2)
