"""Time-series forecasting task tests."""

import torch

from graft_net.tasks.time_series_forecasting import TimeSeriesForecastingModel


def test_time_series_model_predicts_forecast_horizon() -> None:
    model = TimeSeriesForecastingModel(
        embed_dim=32, num_layers=2, num_heads=4, horizon=6, input_features=4
    )
    batch = {"inputs": torch.randn(2, 24, 4), "targets": torch.randn(2, 6, 4)}
    out = model(batch)
    assert out["predictions"].shape == (2, 6, 4)
    assert out["task_loss"].item() >= 0


def test_time_series_backward_passes() -> None:
    model = TimeSeriesForecastingModel(embed_dim=32, num_layers=2, num_heads=4,
                                       horizon=4, input_features=3)
    batch = {"inputs": torch.randn(2, 12, 3), "targets": torch.randn(2, 4, 3)}
    out = model(batch)
    out["task_loss"].backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
