"""Model configuration dataclasses for GRAFT-Net."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraftNetConfig:
    """Central configuration for GRAFT-Net model and all sub-modules."""

    # Core dimensions
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    max_seq_len: int = 512
    dropout: float = 0.1

    # Predictive attention
    predictor_hidden_dim: int = 128
    predictor_depth: int = 1
    use_predictive_attention: bool = True

    # Latent topology
    topology_topk: int = 4
    topology_edge_hidden: int = 64
    use_latent_topology: bool = True

    # Gradient-routed experts
    num_experts: int = 8
    experts_topk: int = 2
    expert_hidden_dim: int = 512
    use_gradient_routing: bool = True

    # Loss weights
    lambda_future: float = 0.1
    lambda_grad: float = 0.05
    lambda_topology: float = 0.01
    lambda_balance: float = 0.01

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    use_amp: bool = False  # disabled on CPU by default

    # Misc
    seed: int = 42
    model_name: str = "graft_net"

    # Per-task overrides (set by task configs)
    task_name: str = "sequence_classification"
    num_classes: int = 10
    forecast_horizon: int = 12
    input_features: int = 7

    def smoke_test_variant(self) -> "GraftNetConfig":
        """Return a tiny variant suitable for unit tests."""
        from dataclasses import replace
        return replace(
            self,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            predictor_hidden_dim=64,
            expert_hidden_dim=64,
            topology_topk=2,
            num_experts=4,
            max_seq_len=16,
        )
