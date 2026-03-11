"""Run the full ablation matrix.

Iterates over the four ablation configs and trains each variant for a
configurable number of epochs. Results are written to
``outputs/ablation/<variant>/metrics.json``.

Usage::

    python scripts/run_ablation.py ablation.num_epochs=5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

ABLATION_OVERRIDES: list[dict[str, object]] = [
    {
        "name": "full_model",
        "use_predictive_attention": True,
        "use_latent_topology": True,
        "use_gradient_routing": True,
    },
    {
        "name": "no_predictive_attention",
        "use_predictive_attention": False,
        "use_latent_topology": True,
        "use_gradient_routing": True,
    },
    {
        "name": "no_latent_topology",
        "use_predictive_attention": True,
        "use_latent_topology": False,
        "use_gradient_routing": True,
    },
    {
        "name": "no_gradient_routing",
        "use_predictive_attention": True,
        "use_latent_topology": True,
        "use_gradient_routing": False,
    },
    {
        "name": "no_topology_no_routing",
        "use_predictive_attention": True,
        "use_latent_topology": False,
        "use_gradient_routing": False,
    },
]


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from dataclasses import replace

    from graft_net.models.config import GraftNetConfig
    from graft_net.train.trainer import Trainer
    from graft_net.utils.seeding import set_seed

    num_epochs = int(OmegaConf.select(cfg, "ablation.num_epochs", default=3))
    task_name = cfg.task.name
    seed = int(cfg.model.get("seed", 42))

    results: dict[str, dict] = {}

    for variant in ABLATION_OVERRIDES:
        name = str(variant["name"])
        set_seed(seed)
        log.info("Running ablation: %s", name)

        output_dir = Path("outputs/ablation") / name
        trainer = Trainer.for_smoke_test(
            output_dir=output_dir,
            task_name=task_name,
            num_classes=int(cfg.task.get("num_classes", 10)),
            seed=seed,
        )

        # Patch ablation flags on trainer.model.cfg (best-effort)
        model_cfg = trainer.cfg
        patched = replace(
            model_cfg,
            use_predictive_attention=bool(variant["use_predictive_attention"]),
            use_latent_topology=bool(variant["use_latent_topology"]),
            use_gradient_routing=bool(variant["use_gradient_routing"]),
        )
        trainer.cfg = patched

        history = trainer.train(num_epochs=num_epochs)
        final_val_loss = history["val/task_loss"][-1]
        results[name] = {"val_loss": final_val_loss, "history": history}
        log.info("  %s final val_loss: %.4f", name, final_val_loss)

        # Persist
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps({"val_loss": final_val_loss}, indent=2)
        )

    # Summary table
    log.info("\n--- Ablation Summary ---")
    for name, r in results.items():
        log.info("  %-35s val_loss=%.4f", name, r["val_loss"])


if __name__ == "__main__":
    main()
