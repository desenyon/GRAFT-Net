"""Build each ablation before training and save the actual resolved configuration."""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from graft_net.experiments import run_experiment

ABLATION_OVERRIDES = [
    {"name": "full_model"},
    {"name": "no_predictive_attention", "use_predictive_attention": False},
    {"name": "no_latent_topology", "use_latent_topology": False},
    {"name": "no_gradient_routing", "use_gradient_routing": False},
    {"name": "no_topology_no_routing", "use_latent_topology": False, "use_gradient_routing": False},
]


def run_ablation(cfg: DictConfig) -> dict:
    output_dir = Path(cfg.output_dir)
    results = {}
    for variant in ABLATION_OVERRIDES:
        name = str(variant["name"])
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        run_cfg.model.model_name = "graft_net"
        for flag in ("use_predictive_attention", "use_latent_topology", "use_gradient_routing"):
            run_cfg.model[flag] = variant.get(flag, True)
        run_cfg.output_dir = str(output_dir / name)
        results[name] = run_experiment(run_cfg, int(cfg.ablation.num_epochs), output_dir / name)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_ablation(cfg)


if __name__ == "__main__":
    main()
