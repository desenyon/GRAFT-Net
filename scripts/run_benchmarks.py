"""Run registered backbones on actual task datasets; no claimed budget matching."""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from graft_net.experiments import TASK_NAMES, run_experiment
from graft_net.models.registry import MODEL_NAMES

MODELS = list(MODEL_NAMES)
TASKS = list(TASK_NAMES)


def run_benchmarks(cfg: DictConfig) -> dict:
    models, tasks = list(cfg.benchmark.models), list(cfg.benchmark.tasks)
    if not models or any(m not in (*MODEL_NAMES, "standard_transformer") for m in models):
        raise ValueError(f"benchmark.models must select from {MODEL_NAMES}")
    if not tasks or any(t not in TASK_NAMES for t in tasks):
        raise ValueError(f"benchmark.tasks must select from {TASK_NAMES}")
    output_dir = Path(cfg.output_dir)
    results = {}
    for task in tasks:
        results[task] = {}
        for model in models:
            run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            if task != cfg.task.name:
                run_cfg.task = OmegaConf.load(
                    Path(__file__).resolve().parents[1] / "configs" / "task" / f"{task}.yaml"
                )
            run_cfg.model.model_name = model
            run_cfg.output_dir = str(output_dir / task / model)
            results[task][model] = run_experiment(
                run_cfg, int(cfg.benchmark.num_epochs), output_dir / task / model
            )
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    rows = [
        "# Synthetic smoke comparisons",
        "",
        "Single-seed losses; parameter/FLOP budgets are not matched.",
        "",
        "| Task | " + " | ".join(models) + " |",
        "| --- | " + " | ".join("---" for _ in models) + " |",
    ]
    for task in tasks:
        losses = " | ".join(
            f"{results[task][model]['metrics']['val/task_loss']:.6f}" for model in models
        )
        rows.append(f"| {task} | {losses} |")
    (output_dir / "results.md").write_text("\n".join(rows) + "\n")
    return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_benchmarks(cfg)


if __name__ == "__main__":
    main()
