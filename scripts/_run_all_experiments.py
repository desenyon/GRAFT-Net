"""Legacy combined entrypoint, using the same verified public sweep implementations.

Unlike v0.1.0, importing this module never starts experiments, and baselines are
registered architectures rather than renamed ablations.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from run_ablation import run_ablation
    from run_benchmarks import run_benchmarks

    output = Path(cfg.output_dir)
    for name, runner in (("ablation", run_ablation), ("benchmarks", run_benchmarks)):
        config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        config.output_dir = str(output / name)
        runner(config)


if __name__ == "__main__":
    main()
