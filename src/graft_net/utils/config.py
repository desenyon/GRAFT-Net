"""Hydra config utilities."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def pretty_print_config(cfg: DictConfig) -> None:
    """Print the resolved Hydra config."""
    print(OmegaConf.to_yaml(cfg))


def config_to_flat_dict(cfg: DictConfig) -> dict:
    """Flatten a nested OmegaConf config to a flat dict for MLflow logging."""
    flat: dict = {}
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise ValueError("Expected a configuration mapping")
    for key, val in values.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat[str(key) + "." + str(subkey)] = subval
        else:
            flat[key] = val
    return flat
