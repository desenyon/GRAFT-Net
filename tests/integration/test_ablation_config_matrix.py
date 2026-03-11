"""Integration test: ablation config matrix loads cleanly via Hydra compose."""

import pytest

pytest.importorskip("hydra")

from hydra import compose, initialize_config_dir
from pathlib import Path

CONFIG_DIR = str(Path(__file__).parent.parent.parent / "configs")

ABLATION_OVERRIDES = [
    ["model=graft_net", "model.use_predictive_attention=false"],
    ["model=graft_net", "model.use_latent_topology=false"],
    ["model=graft_net", "model.use_gradient_routing=false"],
    ["model=graft_net", "model.use_predictive_attention=false", "model.use_latent_topology=false", "model.use_gradient_routing=false"],
]

@pytest.mark.parametrize("overrides", ABLATION_OVERRIDES)
def test_ablation_config_loads(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose("config", overrides=overrides)
        assert hasattr(cfg, "model")
