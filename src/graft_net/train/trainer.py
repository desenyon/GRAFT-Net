"""Shared task trainer with explicit tracking, reproducible checkpoints, and AMP."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from graft_net.eval.evaluator import evaluate
from graft_net.losses.total import compute_total_loss
from graft_net.models.config import GraftNetConfig
from graft_net.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: GraftNetConfig,
        output_dir: Path,
        *,
        device: str = "auto",
        experiment_config: DictConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_config = experiment_config
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda, mps, or auto")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.optimizer = AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.scheduler = LambdaLR(
            self.optimizer, lambda step: min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        )
        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._step = 0
        self._epoch = 0
        self.history: dict[str, list[float]] = {}
        self._mlflow_run_id: str | None = None
        if experiment_config is not None:
            OmegaConf.save(experiment_config, self.output_dir / "config.yaml", resolve=True)
        (self.output_dir / "model_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    @classmethod
    def from_config(cls, config: DictConfig, output_dir: Path | None = None) -> Trainer:
        from graft_net.experiments import build_trainer

        return build_trainer(config, output_dir)

    @classmethod
    def for_smoke_test(
        cls,
        output_dir: Path | str = "/tmp/graft_smoke",  # noqa: S108 - legacy smoke API default
        task_name: str = "sequence_classification",
        num_classes: int = 4,
        seed: int = 42,
    ) -> Trainer:
        cfg = replace(
            GraftNetConfig().smoke_test_variant(),
            task_name=task_name,
            num_classes=num_classes,
            seed=seed,
            warmup_steps=0,
        )
        config = OmegaConf.create(
            {
                "model": asdict(cfg),
                "output_dir": str(output_dir),
                "compute": {"batch_size": 8, "num_workers": 0, "num_threads": 1, "device": "cpu"},
                "task": {
                    "name": task_name,
                    "dataset": "synthetic",
                    "num_train_samples": 24,
                    "num_val_samples": 8,
                    "num_classes": num_classes,
                    "seq_len": 8,
                    "context_len": 8,
                    "num_nodes": 8,
                    "horizon": 4,
                    "input_features": 3,
                    "embed_dim_nodes": cfg.embed_dim,
                    "edge_prob": 0.3,
                },
                "tracking": {"enabled": False},
            }
        )
        return cls.from_config(config)

    def run_smoke_epoch(self) -> dict[str, float]:
        return {**self._train_epoch(), **self.evaluate()}

    def train(self, num_epochs: int = 10) -> dict[str, list[float]]:
        """Train additional epochs; MLflow is opt-in and never silently required."""
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        tracking = self.experiment_config.get("tracking", {}) if self.experiment_config else {}
        if tracking.get("enabled", False):
            try:
                import mlflow
            except ImportError as exc:
                raise RuntimeError("Install graft-net[tracking] to enable MLflow") from exc
            if tracking.get("uri"):
                mlflow.set_tracking_uri(tracking["uri"])
            mlflow.set_experiment(tracking.get("experiment", "graft-net"))
            with mlflow.start_run() as run:
                self._mlflow_run_id = run.info.run_id
                mlflow.log_params(asdict(self.cfg))
                self._run_epochs(num_epochs, tracker=mlflow)
        else:
            self._run_epochs(num_epochs)
        return self.history

    def _run_epochs(self, num_epochs: int, tracker: Any = None) -> None:
        for _ in range(num_epochs):
            metrics = {**self._train_epoch(), **self.evaluate()}
            self._epoch += 1
            for name, value in metrics.items():
                self.history.setdefault(name, []).append(value)
            if tracker is not None:
                tracker.log_metrics(metrics, step=self._step)
            (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2))
            logger.info("Epoch %s: %s", self._epoch, metrics)

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        count = 0
        start = time.perf_counter()
        for batch in self.train_loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            size = next(iter(batch.values())).shape[0]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(batch)
                losses = compute_total_loss(
                    outputs,
                    lambda_future=self.cfg.lambda_future,
                    lambda_grad=self.cfg.lambda_grad,
                    lambda_topology=self.cfg.lambda_topology,
                    lambda_balance=self.cfg.lambda_balance,
                )
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {self._step}")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item() * size
            count += size
            self._step += 1
        if count == 0:
            raise ValueError("Training loader is empty")
        metrics = {f"train/{name}_loss": value / count for name, value in totals.items()}
        metrics["train/throughput"] = count / max(time.perf_counter() - start, 1e-9)
        metrics["train/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    def evaluate(self) -> dict[str, float]:
        return {
            key.replace("eval/", "val/", 1): value
            for key, value in evaluate(self.model, self.val_loader, self.device).items()
        }

    def _eval_epoch(self) -> dict[str, float]:
        """Compatibility alias; new callers should use evaluate()."""
        return self.evaluate()

    def save_checkpoint(self, path: Path | None = None) -> Path:
        path = Path(path) if path else self.output_dir / f"checkpoint_step{self._step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        config = (
            OmegaConf.to_container(self.experiment_config, resolve=True)
            if self.experiment_config
            else None
        )
        payload = {
            "format_version": 2,
            "model_state": self.model.state_dict(),
            "model_config": asdict(self.cfg),
            "experiment_config": config,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "step": self._step,
            "epoch": self._epoch,
            "history": self.history,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "loader_rng_state": self.train_loader.generator.get_state()
            if self.train_loader.generator
            else None,
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if self.device.type == "cuda"
            else None,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def load_checkpoint(self, path: Path, *, restore_rng: bool = True) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "model_config" in ckpt and ckpt["model_config"] != asdict(self.cfg):
            raise ValueError("Checkpoint model config differs; use Trainer.from_checkpoint()")
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self._step, self._epoch = ckpt.get("step", 0), ckpt.get("epoch", 0)
        self.history = ckpt.get("history", {})
        if restore_rng and "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
            random.setstate(ckpt["python_rng_state"])
            if self.train_loader.generator is not None and ckpt.get("loader_rng_state") is not None:
                self.train_loader.generator.set_state(ckpt["loader_rng_state"])
            if self.device.type == "cuda" and ckpt.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])

    @classmethod
    def from_checkpoint(
        cls, path: Path, *, output_dir: Path | None = None, device: str = "cpu"
    ) -> Trainer:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if not ckpt.get("experiment_config"):
            raise ValueError(
                "Checkpoint lacks experiment config; construct a matching Trainer "
                "and call load_checkpoint"
            )
        config = OmegaConf.create(ckpt["experiment_config"])
        config.compute.device = device
        config.output_dir = str(output_dir or Path(path).parent / "evaluation")
        trainer = cls.from_config(config)
        trainer.load_checkpoint(path)
        return trainer
