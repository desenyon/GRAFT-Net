"""Train the configured task/model, or resume a metadata-rich checkpoint."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from graft_net.experiments import run_experiment
from graft_net.train.trainer import Trainer

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    if cfg.resume_from:
        trainer = Trainer.from_checkpoint(
            Path(cfg.resume_from), output_dir=output_dir, device=cfg.compute.device
        )
        trainer.train(int(cfg.training.num_epochs))
        checkpoint = trainer.save_checkpoint()
    else:
        result = run_experiment(cfg, int(cfg.training.num_epochs), output_dir)
        checkpoint = result["checkpoint"]
    log.info("Checkpoint saved: %s", checkpoint)


if __name__ == "__main__":
    main()
