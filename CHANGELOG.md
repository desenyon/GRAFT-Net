# Changelog

All notable changes to GRAFT-Net are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-03-11

### Added

#### Architecture
- `PredictiveAttention` module: multi-head attention with queries computed from a predicted future latent state (H1).
- `LatentTopologyModule`: soft adjacency edge scoring, top-k sparsification, and degree-normalized message passing (H3).
- `GradientRoutedExperts`: Mixture-of-Experts with gradient-utility-supervised routing (H2).
- `GraftBlock`: full GRAFT layer composing all three mechanisms with residual connections and layer norms.
- `GraftNetBackbone`: N-layer stacked GraftBlock with positional embeddings.
- Task heads: `SequenceClassificationHead`, `ForecastingHead`, `GraphClassificationHead`.

#### Baseline Models
- `StandardTransformer`: vanilla `nn.TransformerEncoder` baseline.
- `MoETransformer`: MoE baseline with similarity routing (no gradient supervision).
- `GraphTransformer`: explicit adjacency + message passing baseline.
- `build_model()` registry for all models.

#### Task Models
- `SequenceClassificationModel` (H1 + H2 + H3 full pipeline).
- `TimeSeriesForecastingModel` with input projection.
- `GraphPredictionModel` (node-feature-as-sequence).

#### Losses
- `future_prediction_loss`: MSE between predicted and ground-truth future states.
- `gradient_prediction_loss`: MSE between predicted and actual gradient norms.
- `topology_entropy_loss`: penalises over-concentrated adjacency distributions.
- `load_balance_loss`: squared deviation from uniform expert load.
- `compute_total_loss()`: weighted combination of all losses.

#### Training & Evaluation
- `Trainer` with AMP, gradient clipping, MLflow tracking, and checkpointing.
- `Trainer.for_smoke_test()` factory for fast integration testing.
- `Evaluator` class for inference-only evaluation.

#### Configuration (Hydra)
- Root `configs/config.yaml` with model / compute / task defaults.
- `configs/model/graft_net.yaml`: full model hyperparameters.
- `configs/compute/local.yaml` and `scale.yaml`.
- `configs/task/sequence_classification.yaml`, `time_series_forecasting.yaml`, `graph_prediction.yaml`.
- `configs/ablation/no_predictive_attention.yaml`, `no_latent_topology.yaml`, `no_gradient_routing.yaml`, `no_topology_no_routing.yaml`.
- `configs/baseline/transformer.yaml`, `moe_transformer.yaml`, `graph_transformer.yaml`.

#### Visualization
- `plot_training_curves` — loss/metric line plots.
- `plot_topology_graph` — networkx adjacency visualisation.
- `plot_routing_heatmap` — expert routing score imshow.
- `plot_gradient_utility` — predicted vs actual gradient utility scatter.

#### Scripts
- `scripts/train.py` — Hydra-powered CLI training entry point.
- `scripts/evaluate.py` — evaluation entry point with checkpoint loading.
- `scripts/run_ablation.py` — automated 5-variant ablation matrix.
- `scripts/run_benchmarks.py` — 4-model × 3-task benchmark table.
- `scripts/generate_figures.py` — all paper figures.

#### CI / CD
- GitHub Actions workflow: lint (ruff + mypy), test suite (pytest --cov), smoke tests.
- `Dockerfile` for reproducible containerised execution.

#### Tests
- 16 unit test modules (one per source module).
- 3 integration tests: ablation config matrix, end-to-end smoke, viz output.
- Coverage threshold: 75%.

#### Documentation
- `docs/plans/2026-03-11-graft-net-v0.1.0-master-plan.md` — comprehensive 18-skill master plan.
- `docs/reproducibility.md` — exact reproduction commands.
- `README.md` — quickstart, architecture overview, hypothesis table.

---

[0.1.0]: https://github.com/your-org/GRAFT-Net/releases/tag/v0.1.0
