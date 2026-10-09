# Reproducibility and experiment evidence

The supported reproducibility claim is that the code and small synthetic pipelines can be executed and tested. No expected performance table is supplied. The previous illustrative loss table and the historical `RESEARCH.md` do not validate the corrected mechanisms or baseline comparisons.

## Environment

Use Python 3.11+ in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m pip freeze > outputs-environment.txt
```

`requirements.txt` pins direct dependencies, not every transitive dependency or platform-specific wheel. Editable installation should keep compatible pins, but always record the final environment. CPU validation is the reference; this project does not promise identical results across PyTorch versions, devices or operating systems.

## Offline verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg pytest \
  --cov=graft_net --cov-report=term-missing --cov-fail-under=75
ruff check src tests scripts
ruff format --check src tests scripts
mypy src/graft_net --ignore-missing-imports
python -m build
```

The suite uses generated tensors and temporary directories. MLflow is off unless explicitly enabled. The CLI regression tests construct all three task datasets and all four registered backbones, run actual optimization, and compare reconstructed evaluation results.

## Reproduce a smoke run and evaluation

```bash
python scripts/train.py compute=smoke seed=42 output_dir=outputs/repro
python scripts/evaluate.py checkpoint=outputs/repro/checkpoint_step3.pt \
  output_dir=outputs/repro-evaluation
python scripts/train.py compute=smoke resume_from=outputs/repro/checkpoint_step3.pt \
  training.num_epochs=1 output_dir=outputs/repro-resume
```

The smoke profile has three training batches per epoch. Other batch/sample settings produce different checkpoint step numbers; use the path printed at the end of the run.

`config.yaml`, `model_config.json`, `history.json`, and `metrics.json` identify what actually ran. For ordinary runs, metadata includes the Git revision and dirty status, actual backbone class, task, seed, parameter count, Python/PyTorch version, and device. The checkpoint stores the resolved experiment configuration and epoch-boundary state. A direct custom `Trainer` without experiment metadata still requires explicit model and loader reconstruction.

## Seeds, splits and continuation

- The root seed defaults to 42 and feeds the model seed; `model.seed` can be explicitly overridden.
- Python and PyTorch RNG are initialized before construction. Synthetic data uses its own PyTorch generator.
- A single generated population shares the same class offsets, then disjoint train and validation subsets are selected using `seed + 1`.
- A separate `seed + 2` generator shuffles training data. Architecture-specific initialization cannot alter its sample order.
- Validation is not shuffled and uses a separate generator.
- Format-v2 checkpoints save optimizer, scheduler, scaler, epoch, step, history and relevant RNG states. The regression suite checks exact CPU parameter equality after uninterrupted versus resumed epochs.
- Guarantees do not extend to mid-epoch restart, distributed training, custom stochastic data-worker state, device changes, or library upgrades.

## Comparisons

```bash
python scripts/run_ablation.py compute=smoke ablation.num_epochs=1 \
  output_dir=outputs/repro-ablations
python scripts/run_benchmarks.py compute=smoke benchmark.num_epochs=1 \
  output_dir=outputs/repro-benchmarks
```

Sweep results include configuration and per-run evidence. Increase budgets and repeat with explicit seeds only after smoke checks. Parameter/FLOP matching, real datasets, confidence intervals and external held-out test evaluation are not implemented automatically. The `no_gradient_routing` control changes the expert architecture to a dense FFN; `lambda_grad=0` is the narrower supervision-only control.

Plot recorded history with:

```bash
python scripts/generate_figures.py --history outputs/repro/history.json \
  --output-dir outputs/repro/figures
```

`--demo` generates clearly labeled artificial examples and must not be cited as experiment evidence.
