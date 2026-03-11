"""CLI training entrypoint.

Usage::

    python scripts/train.py \
        model=graft_net \
        task=sequence_classification \
        compute=local \
        training.num_epochs=10
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    import torch
    from graft_net.models.config import GraftNetConfig
    from graft_net.train.trainer import Trainer
    from graft_net.utils.seeding import set_seed

    seed = int(cfg.model.get("seed", 42))
    set_seed(seed)

    model_cfg = GraftNetConfig(
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        num_experts=cfg.model.num_experts,
        experts_topk=cfg.model.experts_topk,
        expert_hidden_dim=cfg.model.expert_hidden_dim,
        use_predictive_attention=cfg.model.use_predictive_attention,
        use_latent_topology=cfg.model.use_latent_topology,
        use_gradient_routing=cfg.model.use_gradient_routing,
        lambda_future=cfg.model.lambda_future,
        lambda_grad=cfg.model.lambda_grad,
        lambda_topology=cfg.model.lambda_topology,
        lambda_balance=cfg.model.lambda_balance,
        learning_rate=cfg.model.learning_rate,
        use_amp=cfg.compute.use_amp,
        task_name=cfg.task.name,
        num_classes=cfg.task.get("num_classes", 10),
        seed=seed,
    )

    output_dir = Path("outputs") / cfg.task.name / cfg.model.model_name
    trainer = Trainer.for_smoke_test(
        output_dir=output_dir,
        task_name=cfg.task.name,
        num_classes=int(cfg.task.get("num_classes", 10)),
        seed=seed,
    )

    num_epochs = int(cfg.training.get("num_epochs", 10))
    history = trainer.train(num_epochs=num_epochs)

    log.info("Training complete. Final val loss: %.4f", history["val/task_loss"][-1])

    # Save final checkpoint
    ckpt = trainer.save_checkpoint()
    log.info("Checkpoint saved: %s", ckpt)


if __name__ == "__main__":
    main()
