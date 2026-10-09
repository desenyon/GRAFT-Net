# Upgrade validation record

This records functional checks for the experiment/gradient upgrade, not model-quality results.

## Baseline

The starting revision was `fac77abcb4a166bdb05f25538c54b874cd86859c`.

- 47 existing tests passed; coverage was 83.95%.
- The then-installed Ruff reported 105 lint findings and 35 files needing formatting.
- The then-installed mypy reported 13 errors in configuration/registry code.
- Added regressions reproduced incorrect task dispatch, ignored configuration, absent topology task gradients, self-targeted router supervision, ineffective ablation losses, and padding leakage before fixes.

## Final local checks

Reference environment: macOS arm64, Python 3.11.14, PyTorch 2.10.0, Hydra 1.3.2, OmegaConf 2.3.0, pytest 8.4.1, Ruff 0.15.1, mypy 1.19.1. Tests used one CPU thread and generated data.

| Check | Observed result |
| --- | --- |
| Full pytest suite with coverage | 94 passed; 91.75% source coverage |
| `ruff check src tests scripts` | Passed |
| `ruff format --check src tests scripts` | Passed, 96 files |
| `mypy src/graft_net --ignore-missing-imports` | Passed, 53 source files |
| `python -m pip check` | No broken requirements |
| `python -m build --no-isolation` | Wheel and source distribution built |
| Import from built wheel and forecasting smoke epoch | Passed |
| Extracted source distribution's train CLI | Passed; configs/scripts present and checkpoint written |
| Recorded-history figure CLI | Passed in subprocess regression |
| Explicit demo figures | All four labeled demo files generated |
| Optional MLflow 3.10.1 with local SQLite | Finished run; stored seed/config parameters and epoch metrics verified through the client |
| `git diff --check` | Passed |

The integration suite exercises all 12 combinations of task and registered backbone, actual training/evaluation/resume CLI commands, all five ablations and the full benchmark matrix. Assertions check the instantiated classes and saved configuration, not just that a loss is positive.

Gradient tests verify nonzero task-path scorer gradients at topology k=1 and k=2, detached label-dependent routing targets and auxiliary gradients in every block. Further tests cover exact epoch-boundary CPU resume, deterministic data order independent of architecture initialization, sample-weighted uneven batches, disabled auxiliary losses, mask invariance, legacy registry FFN widths, absent Git executables, and half-precision masked topology diagnostics.

## Limits

- No expensive training, real-data accuracy comparison, throughput comparison, multi-seed research study, CUDA AMP or MPS validation was performed.
- The Dockerfile was corrected to include the README required by package metadata, but Docker image build/execution was not run: the local Docker daemon was unavailable.
- CI is configured to use pinned direct dependencies and CPU PyTorch, and to run source checks, coverage, integration smoke tests and package builds. Its result for the pushed revision is reported in the draft PR, rather than assumed here.
