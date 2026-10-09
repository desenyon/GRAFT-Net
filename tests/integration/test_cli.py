"""Offline subprocess coverage of the public experiment entrypoints."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def run_cli(script, *args):
    result = subprocess.run(  # noqa: S603 - fixed Python CLI with test-generated arguments
        [sys.executable, str(ROOT / "scripts" / script), "compute=smoke", *args],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_train_evaluate_and_resume_saved_configuration(tmp_path):
    output = tmp_path / "train"
    run_cli(
        "train.py",
        f"output_dir={output}",
        "task=graph_prediction",
        "model.model_name=graph_transformer",
        "model.embed_dim=16",
        "model.num_heads=2",
        "task.embed_dim_nodes=7",
    )
    checkpoints = list(output.glob("checkpoint_step*.pt"))
    assert len(checkpoints) == 1
    saved = torch.load(checkpoints[0], weights_only=True)
    assert saved["model_config"]["embed_dim"] == 16
    assert saved["model_config"]["node_input_dim"] == 7
    evaluation = tmp_path / "eval"
    run_cli("evaluate.py", f"checkpoint={checkpoints[0]}", f"output_dir={evaluation}")
    metrics = json.loads((evaluation / "metrics.json").read_text())
    history = json.loads((output / "history.json").read_text())
    assert metrics["val/task_loss"] == history["val/task_loss"][-1]
    resume = tmp_path / "resume"
    run_cli("train.py", f"resume_from={checkpoints[0]}", f"output_dir={resume}")
    continued = torch.load(next(resume.glob("checkpoint_step*.pt")), weights_only=True)
    assert continued["step"] == 2 * saved["step"]


def test_sweeps_record_real_variants_and_models(tmp_path):
    ablation = tmp_path / "ablations"
    run_cli("run_ablation.py", f"output_dir={ablation}", "ablation.num_epochs=1")
    variants = json.loads((ablation / "results.json").read_text())
    assert len(variants) == 5
    for variant in ("no_predictive_attention", "no_latent_topology", "no_gradient_routing"):
        flag = variant.replace("no_", "use_")
        saved = json.loads((ablation / variant / "model_config.json").read_text())
        assert saved[flag] is False
    benchmark = tmp_path / "benchmarks"
    run_cli(
        "run_benchmarks.py",
        f"output_dir={benchmark}",
        "benchmark.num_epochs=1",
        "benchmark.tasks=[sequence_classification,time_series_forecasting,graph_prediction]",
    )
    results = json.loads((benchmark / "results.json").read_text())
    assert len(results) == 3
    for task, models in results.items():
        assert len(models) == 4
        for model, result in models.items():
            saved = json.loads((benchmark / task / model / "model_config.json").read_text())
            assert saved["model_name"] == model
            assert result["parameters"] > 0
            assert result["metrics"]["val/task_loss"] >= 0
