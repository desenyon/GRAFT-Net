"""Benchmark GRAFT-Net against all baselines on all tasks.

Outputs a Markdown table to ``outputs/benchmarks/results.md``.

Usage::

    python scripts/run_benchmarks.py benchmark.num_epochs=5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

MODELS = ["graft_net", "standard_transformer", "moe_transformer", "graph_transformer"]
TASKS = ["sequence_classification", "time_series_forecasting", "graph_prediction"]


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from graft_net.train.trainer import Trainer
    from graft_net.utils.seeding import set_seed

    num_epochs = int(OmegaConf.select(cfg, "benchmark.num_epochs", default=3))
    num_classes = int(OmegaConf.select(cfg, "task.num_classes", default=10))
    seed = int(cfg.model.get("seed", 42))

    results: dict[str, dict[str, float]] = {}

    for task in TASKS:
        results[task] = {}
        for model_name in MODELS:
            set_seed(seed)
            log.info("Benchmarking  model=%s  task=%s", model_name, task)
            output_dir = Path("outputs/benchmarks") / task / model_name
            trainer = Trainer.for_smoke_test(
                output_dir=output_dir,
                task_name=task,
                num_classes=num_classes,
                seed=seed,
            )
            history = trainer.train(num_epochs=num_epochs)
            val_loss = history["val/task_loss"][-1]
            results[task][model_name] = val_loss
            log.info("    val_loss=%.4f", val_loss)

    # Write JSON
    out_dir = Path("outputs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Write Markdown table
    header = "| Task | " + " | ".join(MODELS) + " |"
    sep = "| --- " * (len(MODELS) + 1) + "|"
    rows = [header, sep]
    for task in TASKS:
        row_vals = " | ".join(f"{results[task].get(m, float('nan')):.4f}" for m in MODELS)
        rows.append(f"| {task} | {row_vals} |")
    md = "\n".join(rows)
    (out_dir / "results.md").write_text(md)
    log.info("\n%s", md)


if __name__ == "__main__":
    main()
