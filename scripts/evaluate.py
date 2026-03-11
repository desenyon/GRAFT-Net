"""CLI evaluation entrypoint.

Usage::

    python scripts/evaluate.py \
        task=sequence_classification \
        checkpoint=outputs/sequence_classification/graft_net/checkpoint_step80.pt
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from graft_net.eval.evaluator import Evaluator
    from graft_net.models.config import GraftNetConfig
    from graft_net.train.trainer import Trainer
    from graft_net.utils.seeding import set_seed

    set_seed(int(cfg.model.get("seed", 42)))

    output_dir = Path("outputs") / cfg.task.name / cfg.model.model_name
    trainer = Trainer.for_smoke_test(
        output_dir=output_dir,
        task_name=cfg.task.name,
        num_classes=int(cfg.task.get("num_classes", 10)),
    )

    ckpt_path = cfg.get("checkpoint", None)
    if ckpt_path:
        trainer.load_checkpoint(Path(ckpt_path))
        log.info("Loaded checkpoint: %s", ckpt_path)

    metrics = trainer._eval_epoch()
    for k, v in metrics.items():
        log.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()
