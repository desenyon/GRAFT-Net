"""Shared, deterministic construction for CLIs, sweeps, and small smoke tests."""

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split

from graft_net.data.graph_synthetic import SyntheticGraphDataset
from graft_net.data.sequence_synthetic import SyntheticSequenceDataset
from graft_net.data.time_series_synthetic import SyntheticTimeSeriesDataset
from graft_net.models.config import GraftNetConfig
from graft_net.tasks.graph_prediction import GraphPredictionModel
from graft_net.tasks.sequence_classification import SequenceClassificationModel
from graft_net.tasks.time_series_forecasting import TimeSeriesForecastingModel
from graft_net.utils.seeding import set_seed

if TYPE_CHECKING:
    from graft_net.train.trainer import Trainer

TASK_NAMES = ("sequence_classification", "time_series_forecasting", "graph_prediction")


def resolve_model_config(config: DictConfig) -> GraftNetConfig:
    values = OmegaConf.to_container(config.model, resolve=True)
    if not isinstance(values, dict):
        raise ValueError("model must be a mapping")
    allowed = {field.name for field in fields(GraftNetConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown model settings: {sorted(unknown)}")
    model_values: dict[str, Any] = {str(key): value for key, value in values.items()}
    model_values.update(
        task_name=config.task.name,
        num_classes=config.task.get("num_classes", 10),
        forecast_horizon=config.task.get("horizon", 12),
        input_features=config.task.get("input_features", 7),
        node_input_dim=config.task.get("embed_dim_nodes"),
    )
    return GraftNetConfig(**model_values)


def build_loaders(config: DictConfig, cfg: GraftNetConfig) -> tuple[DataLoader, DataLoader]:
    task = config.task
    if task.dataset != "synthetic":
        raise ValueError("Only dataset=synthetic is implemented")
    n_train, n_val = int(task.num_train_samples), int(task.num_val_samples)
    if min(n_train, n_val, int(config.compute.batch_size)) <= 0:
        raise ValueError("Dataset sizes and batch_size must be positive")
    if int(config.compute.num_workers) < 0:
        raise ValueError("num_workers must be nonnegative")
    length_fields = {
        "sequence_classification": "seq_len",
        "time_series_forecasting": "context_len",
        "graph_prediction": "num_nodes",
    }
    if cfg.task_name not in length_fields:
        raise ValueError(f"Unknown task {cfg.task_name!r}; choose from {TASK_NAMES}")
    length = int(task[length_fields[cfg.task_name]])
    if not 0 < length <= cfg.max_seq_len:
        raise ValueError(
            f"Task length {length} must be positive and <= max_seq_len={cfg.max_seq_len}"
        )
    common: dict[str, Any] = {"num_samples": n_train + n_val, "seed": cfg.seed}
    dataset: Dataset
    if cfg.task_name == "sequence_classification":
        dataset = SyntheticSequenceDataset(
            **common, seq_len=length, embed_dim=cfg.embed_dim, num_classes=cfg.num_classes
        )
    elif cfg.task_name == "time_series_forecasting":
        dataset = SyntheticTimeSeriesDataset(
            **common,
            context_len=length,
            horizon=cfg.forecast_horizon,
            input_features=cfg.input_features,
        )
    else:
        if not 0 <= float(task.edge_prob) <= 1:
            raise ValueError("edge_prob must be in [0, 1]")
        dataset = SyntheticGraphDataset(
            **common,
            num_nodes=length,
            embed_dim=cfg.node_input_dim or cfg.embed_dim,
            num_classes=cfg.num_classes,
            edge_prob=float(task.edge_prob),
        )
    # One generated population shares class offsets; separate datasets with different
    # seeds would accidentally change the label-to-pattern mapping between splits.
    train, val = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed + 1)
    )
    loader_args: dict[str, Any] = {
        "batch_size": int(config.compute.batch_size),
        "num_workers": int(config.compute.num_workers),
    }
    return (
        DataLoader(
            train,
            shuffle=True,
            generator=torch.Generator().manual_seed(cfg.seed + 2),
            **loader_args,
        ),
        DataLoader(val, generator=torch.Generator().manual_seed(cfg.seed + 3), **loader_args),
    )


def build_trainer(config: DictConfig, output_dir: Path | None = None) -> "Trainer":
    from graft_net.train.trainer import Trainer

    cfg = resolve_model_config(config)
    threads = int(config.compute.get("num_threads", 1))
    if threads <= 0:
        raise ValueError("num_threads must be positive")
    torch.set_num_threads(threads)
    set_seed(cfg.seed)
    train, val = build_loaders(config, cfg)
    model_type = {
        "sequence_classification": SequenceClassificationModel,
        "time_series_forecasting": TimeSeriesForecastingModel,
        "graph_prediction": GraphPredictionModel,
    }[cfg.task_name]
    return Trainer(
        model_type(cfg=cfg),
        train,
        val,
        cfg,
        output_dir or Path(config.output_dir),
        device=str(config.compute.get("device", "cpu")),
        experiment_config=config,
    )


def run_experiment(config: DictConfig, num_epochs: int, output_dir: Path) -> dict[str, Any]:
    """Train one actual model and persist reviewable run evidence."""
    import json
    import platform
    import shutil
    import subprocess

    trainer = build_trainer(config, output_dir)
    history = trainer.train(num_epochs)
    checkpoint = trainer.save_checkpoint()
    git = shutil.which("git")
    source_root = Path(__file__).resolve().parents[2]
    revision = (
        subprocess.run(  # noqa: S603 - resolved executable and fixed arguments
            [git, "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True, check=False
        )
        if git
        else None
    )
    dirty = (
        subprocess.run(  # noqa: S603 - resolved executable and fixed arguments
            [git, "status", "--porcelain"],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if git
        else None
    )
    result = {
        "model": trainer.cfg.model_name,
        "task": trainer.cfg.task_name,
        "backbone": type(trainer.model.backbone).__name__,
        "seed": trainer.cfg.seed,
        "epochs": num_epochs,
        "parameters": sum(p.numel() for p in trainer.model.parameters()),
        "metrics": {key: values[-1] for key, values in history.items()},
        "checkpoint": str(checkpoint.resolve()),
        "history": history,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": str(trainer.device),
            "git_commit": revision.stdout.strip()
            if revision is not None and revision.returncode == 0
            else None,
            "git_dirty": bool(dirty.stdout.strip())
            if dirty is not None and dirty.returncode == 0
            else None,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
