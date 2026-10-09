"""Time-Series Forecasting task model."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graft_net.models.config import GraftNetConfig
from graft_net.models.heads import ForecastingHead
from graft_net.models.registry import build_backbone
from graft_net.tasks.auxiliary import auxiliary_outputs


class TimeSeriesForecastingModel(nn.Module):
    """GRAFT-Net backbone + forecasting head.

    Input: (B, T, F) — batch of multivariate time series.
    Output: (B, horizon, F) — forecast.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        horizon: int = 12,
        input_features: int = 7,
        cfg: GraftNetConfig | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            from dataclasses import replace

            cfg = GraftNetConfig()
            cfg = replace(
                cfg,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                forecast_horizon=horizon,
                input_features=input_features,
            )
        self.cfg = cfg
        # Project raw features to embed_dim
        self.input_proj = nn.Linear(cfg.input_features, cfg.embed_dim)
        self.backbone = build_backbone(cfg)
        self.head = ForecastingHead(cfg.embed_dim, cfg.input_features, cfg.forecast_horizon)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        batch keys:
            inputs:  (B, T, F) time-series
            targets: (B, horizon, F)
        """
        x = batch["inputs"]  # (B, T, F)
        targets = batch["targets"]  # (B, horizon, F)

        x_emb = self.input_proj(x)  # (B, T, D)
        backbone_out = self.backbone(x_emb)
        predictions = self.head(backbone_out.hidden)  # (B, horizon, F)
        task_loss = F.mse_loss(predictions, targets)

        return {
            "predictions": predictions,
            "task_loss": task_loss,
            **auxiliary_outputs(backbone_out, task_loss, self.cfg, self.training),
        }
