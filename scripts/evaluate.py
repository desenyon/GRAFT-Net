"""Reconstruct the saved experiment and evaluate its held-out synthetic split."""

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from graft_net.train.trainer import Trainer


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.checkpoint:
        raise ValueError("checkpoint=<path> is required for evaluation")
    output_dir = Path(cfg.output_dir)
    trainer = Trainer.from_checkpoint(
        Path(cfg.checkpoint), output_dir=output_dir, device=cfg.compute.device
    )
    metrics = trainer.evaluate()
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logging.getLogger(__name__).info("Evaluation: %s", metrics)


if __name__ == "__main__":
    main()
