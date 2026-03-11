"""Hydra config utilities."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def pretty_print_config(cfg: DictConfig) -> None:
    """Print the resolved Hydra config."""
    print(OmegaConf.to_yaml(cfg))


def config_to_flat_dict(cfg: DictConfig) -> dict:
    """Flatten a nested OmegaConf config to a flat dict for MLflow logging."""
    flat: dict = {}
    for key, val in OmegaConf.to_container(cfg, resolve=True).items():  # type: ignore[arg-type]
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat[f"{key}.{subkey}"] = subval
        else:
            flat[key] = val
    return flat
