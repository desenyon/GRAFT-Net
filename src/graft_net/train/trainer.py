"""Shared Trainer with MLflow tracking, checkpointing, and AMP support."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from graft_net.data.sequence_synthetic import SyntheticSequenceDataset
from graft_net.losses.total import compute_total_loss
from graft_net.models.config import GraftNetConfig
from graft_net.tasks.sequence_classification import SequenceClassificationModel
from graft_net.utils.logging import get_logger
from graft_net.utils.seeding import set_seed

logger = get_logger(__name__)


class Trainer:
    """Training loop with MLflow logging, gradient clipping, and AMP."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: GraftNetConfig,
        output_dir: Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # AMP scaler (no-op on CPU)
        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self._step = 0
        self._mlflow_run_id: str | None = None

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #

    @classmethod
    def for_smoke_test(
        cls,
        output_dir: Path | str = "/tmp/graft_smoke",
        task_name: str = "sequence_classification",
        num_classes: int = 4,
        seed: int = 42,
    ) -> "Trainer":
        """Build a tiny trainer for integration / smoke tests."""
        set_seed(seed)
        cfg = GraftNetConfig().smoke_test_variant()
        cfg = _replace(cfg, task_name=task_name, num_classes=num_classes, seed=seed)

        dataset = SyntheticSequenceDataset(
            num_samples=32, seq_len=8, embed_dim=cfg.embed_dim, num_classes=num_classes, seed=seed,
        )
        train_ds, val_ds = torch.utils.data.random_split(dataset, [24, 8])
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8)

        model = SequenceClassificationModel(
            embed_dim=cfg.embed_dim, num_layers=cfg.num_layers,
            num_heads=cfg.num_heads, num_classes=num_classes, cfg=cfg,
        )
        return cls(model=model, train_loader=train_loader, val_loader=val_loader,
                   cfg=cfg, output_dir=Path(output_dir))

    # ------------------------------------------------------------------ #
    # Core training loop
    # ------------------------------------------------------------------ #

    def run_smoke_epoch(self) -> dict[str, float]:
        """Run one train + eval epoch; return metric dict."""
        train_metrics = self._train_epoch()
        val_metrics = self._eval_epoch()
        metrics = {**train_metrics, **val_metrics}

        # Try to log to MLflow if available
        try:
            import mlflow
            with mlflow.start_run():
                mlflow.log_metrics(metrics, step=self._step)
        except Exception:
            pass  # MLflow optional for smoke tests

        return metrics

    def train(self, num_epochs: int = 10) -> dict[str, list[float]]:
        """Full training loop with MLflow run."""
        history: dict[str, list[float]] = {}

        try:
            import mlflow
            import subprocess
            try:
                git_hash = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=self.output_dir.parent,
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
            except Exception:
                git_hash = "unknown"

            with mlflow.start_run(tags={"git_commit": git_hash}) as run:
                self._mlflow_run_id = run.info.run_id
                mlflow.log_params({
                    "embed_dim": self.cfg.embed_dim,
                    "num_layers": self.cfg.num_layers,
                    "num_heads": self.cfg.num_heads,
                    "num_experts": self.cfg.num_experts,
                    "use_predictive_attention": self.cfg.use_predictive_attention,
                    "use_latent_topology": self.cfg.use_latent_topology,
                    "use_gradient_routing": self.cfg.use_gradient_routing,
                    "task": self.cfg.task_name,
                    "learning_rate": self.cfg.learning_rate,
                })
                history = self._run_epochs(num_epochs)
        except ImportError:
            history = self._run_epochs(num_epochs)

        return history

    def _run_epochs(self, num_epochs: int) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {}
        for epoch in range(num_epochs):
            train_m = self._train_epoch()
            val_m = self._eval_epoch()
            for k, v in {**train_m, **val_m}.items():
                history.setdefault(k, []).append(v)
            logger.info(f"Epoch {epoch+1}/{num_epochs}: {train_m | val_m}")
        return history

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(batch)
                losses = compute_total_loss(
                    outputs,
                    lambda_future=self.cfg.lambda_future,
                    lambda_grad=self.cfg.lambda_grad,
                    lambda_topology=self.cfg.lambda_topology,
                    lambda_balance=self.cfg.lambda_balance,
                )

            loss = losses["total"]
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self._step += 1

        elapsed = time.time() - t0
        return {
            "train/task_loss": total_loss / max(n_batches, 1),
            "train/throughput": len(self.train_loader.dataset) / elapsed,  # type: ignore[arg-type]
        }

    def _eval_epoch(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch)
                losses = compute_total_loss(outputs)
                total_loss += losses["task"].item()
                n_batches += 1
        return {"val/task_loss": total_loss / max(n_batches, 1)}

    def save_checkpoint(self, path: Path | None = None) -> Path:
        path = path or self.output_dir / f"checkpoint_step{self._step}.pt"
        torch.save({"model_state": self.model.state_dict(), "step": self._step}, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self._step = ckpt.get("step", 0)


def _replace(cfg: GraftNetConfig, **kwargs: Any) -> GraftNetConfig:
    from dataclasses import replace
    return replace(cfg, **kwargs)
