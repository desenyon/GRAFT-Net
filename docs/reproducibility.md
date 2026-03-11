# Reproducibility Guide — GRAFT-Net v0.1.0

This document provides the exact commands to reproduce all results reported for GRAFT-Net v0.1.0.

---

## Environment

```bash
# Python 3.11 required
python --version   # Python 3.11.x

# Install with pinned dev extras
pip install -e ".[dev]"
pip freeze > requirements_frozen.txt

# Verify install
python -c "import graft_net; print(graft_net.__version__)"
```

Alternatively, use Docker:

```bash
docker build -t graft-net:v0.1.0 .
docker run --rm graft-net:v0.1.0
```

---

## Seed Policy

All experiments use `seed=42` by default. The seed is passed through:

1. `GraftNetConfig.seed` → `set_seed(seed)` at start of every run
2. `SyntheticDataset(seed=seed)` for deterministic splits
3. Logged as an MLflow parameter on every run

---

## Ablation Matrix

Run all five ablation variants (full + 4 ablations) for 30 epochs each:

```bash
python scripts/run_ablation.py \
    task=sequence_classification \
    ablation.num_epochs=30 \
    compute=local \
    model.seed=42
```

Results are written to `outputs/ablation/<variant>/metrics.json`.

---

## Baseline Comparisons

```bash
python scripts/run_benchmarks.py \
    benchmark.num_epochs=30 \
    compute=local \
    model.seed=42
```

Results table at `outputs/benchmarks/results.md`.

---

## Paper Figures

After running ablation + benchmarks:

```bash
python scripts/generate_figures.py
```

Figures written to `outputs/figures/`.

---

## Expected Results (Indicative, Synthetic Data)

These numbers are on the synthetic datasets shipped with the package and serve as sanity checks only. Real results require task-specific real datasets.

| Variant | val_loss (seq_cls, 30 epochs) |
|---|---|
| full_model | ≈ 0.15 |
| no_predictive_attention | ≈ 0.19 |
| no_latent_topology | ≈ 0.17 |
| no_gradient_routing | ≈ 0.18 |
| no_topology_no_routing | ≈ 0.22 |

---

## Test Suite

```bash
pytest -v \
    --cov=graft_net \
    --cov-report=term-missing \
    --cov-fail-under=75
```

The test suite must pass with ≥75% coverage before any result is considered reproducible.
