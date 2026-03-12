"""Run ablation matrix and benchmark suite, write results to outputs/.

Key design: models are built *after* crafting the variant cfg so that ablation
flags are baked into the module __init__. Using Trainer.for_smoke_test as a
source of dataset/loaders, but replacing the model before training.
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from graft_net.data.sequence_synthetic import SyntheticSequenceDataset
from graft_net.data.time_series_synthetic import SyntheticTimeSeriesDataset
from graft_net.data.graph_synthetic import SyntheticGraphDataset
from graft_net.models.config import GraftNetConfig
from graft_net.models.backbone import GraftNetBackbone
from graft_net.tasks.sequence_classification import SequenceClassificationModel
from graft_net.tasks.time_series_forecasting import TimeSeriesForecastingModel
from graft_net.tasks.graph_prediction import GraphPredictionModel
from graft_net.train.trainer import Trainer
from graft_net.utils.seeding import set_seed

NUM_CLASSES   = 4
FORECAST_HOR  = 6
INPUT_FEATS   = 4
N_EPOCHS      = 20
SEED          = 42
BATCH         = 8
NUM_SAMPLES   = 64
SEQ_LEN       = 12

VARIANTS = [
    {"name": "full_model",              "use_predictive_attention": True,  "use_latent_topology": True,  "use_gradient_routing": True},
    {"name": "no_predictive_attention", "use_predictive_attention": False, "use_latent_topology": True,  "use_gradient_routing": True},
    {"name": "no_latent_topology",      "use_predictive_attention": True,  "use_latent_topology": False, "use_gradient_routing": True},
    {"name": "no_gradient_routing",     "use_predictive_attention": True,  "use_latent_topology": True,  "use_gradient_routing": False},
    {"name": "no_topology_no_routing",  "use_predictive_attention": True,  "use_latent_topology": False, "use_gradient_routing": False},
]

Path("outputs").mkdir(exist_ok=True)


def base_cfg(seed: int = SEED, **flags) -> GraftNetConfig:
    cfg = GraftNetConfig().smoke_test_variant()
    cfg = replace(cfg, seed=seed, num_classes=NUM_CLASSES,
                  forecast_horizon=FORECAST_HOR, input_features=INPUT_FEATS, **flags)
    return cfg


def make_loaders_seq(seed):
    ds = SyntheticSequenceDataset(NUM_SAMPLES, SEQ_LEN, 32, NUM_CLASSES, seed)
    tr, va = random_split(ds, [int(len(ds)*0.75), len(ds)-int(len(ds)*0.75)])
    return DataLoader(tr, batch_size=BATCH, shuffle=True), DataLoader(va, batch_size=BATCH)


def make_loaders_ts(seed):
    ds = SyntheticTimeSeriesDataset(NUM_SAMPLES, SEQ_LEN, FORECAST_HOR, INPUT_FEATS, seed)
    tr, va = random_split(ds, [int(len(ds)*0.75), len(ds)-int(len(ds)*0.75)])
    return DataLoader(tr, batch_size=BATCH, shuffle=True), DataLoader(va, batch_size=BATCH)


def make_loaders_graph(seed):
    ds = SyntheticGraphDataset(NUM_SAMPLES, 8, 32, NUM_CLASSES, seed=seed)
    tr, va = random_split(ds, [int(len(ds)*0.75), len(ds)-int(len(ds)*0.75)])
    return DataLoader(tr, batch_size=BATCH, shuffle=True), DataLoader(va, batch_size=BATCH)


def build_model(task: str, cfg: GraftNetConfig) -> torch.nn.Module:
    if task == "sequence_classification":
        return SequenceClassificationModel(cfg.embed_dim, cfg.num_layers, cfg.num_heads, NUM_CLASSES, cfg)
    elif task == "time_series_forecasting":
        return TimeSeriesForecastingModel(cfg.embed_dim, cfg.num_layers, cfg.num_heads,
                                          horizon=FORECAST_HOR, input_features=INPUT_FEATS, cfg=cfg)
    else:
        return GraphPredictionModel(cfg.embed_dim, cfg.num_layers, cfg.num_heads, NUM_CLASSES, cfg)


def make_loaders(task: str, seed: int):
    if task == "sequence_classification":   return make_loaders_seq(seed)
    elif task == "time_series_forecasting": return make_loaders_ts(seed)
    else:                                   return make_loaders_graph(seed)


def run_variant(task, cfg, seed, out_dir):
    set_seed(seed)
    train_loader, val_loader = make_loaders(task, seed)
    model = build_model(task, cfg)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      cfg=cfg, output_dir=Path(out_dir))
    t0 = time.time()
    history = trainer.train(num_epochs=N_EPOCHS)
    elapsed = time.time() - t0
    val_losses   = history["val/task_loss"]
    train_losses = history["train/task_loss"]
    return {
        "final_val_loss":   round(val_losses[-1], 4),
        "final_train_loss": round(train_losses[-1], 4),
        "best_val_loss":    round(min(val_losses), 4),
        "val_loss_curve":   [round(x, 4) for x in val_losses],
        "train_loss_curve": [round(x, 4) for x in train_losses],
        "epochs": N_EPOCHS,
        "time_s": round(elapsed, 1),
    }


TASKS = ["sequence_classification", "time_series_forecasting", "graph_prediction"]

# ── Ablation ──────────────────────────────────────────────────────────────────
print("\n====== ABLATION MATRIX ======")
ablation_results = {}

for task in TASKS:
    ablation_results[task] = {}
    print(f"\n-- Task: {task} --")
    for v in VARIANTS:
        cfg = base_cfg(
            use_predictive_attention=v["use_predictive_attention"],
            use_latent_topology=v["use_latent_topology"],
            use_gradient_routing=v["use_gradient_routing"],
        )
        r = run_variant(task, cfg, SEED, f"/tmp/graft_ablation/{task}/{v['name']}")
        ablation_results[task][v["name"]] = r
        print(f"  {v['name']:<32} val={r['final_val_loss']:.4f}  best={r['best_val_loss']:.4f}  {r['time_s']:.1f}s")

Path("outputs/ablation_results.json").write_text(json.dumps(ablation_results, indent=2))
print("\nSaved outputs/ablation_results.json")

# ── Benchmarks: baselines are approximated by ablation combos + the full model ─
print("\n====== BASELINE BENCHMARKS ======")
benchmark_results = {}

baseline_variant_map = {
    "graft_net":             dict(use_predictive_attention=True,  use_latent_topology=True,  use_gradient_routing=True),
    "standard_transformer":  dict(use_predictive_attention=False, use_latent_topology=False, use_gradient_routing=False),
    "moe_transformer":       dict(use_predictive_attention=False, use_latent_topology=False, use_gradient_routing=True),
    "graph_transformer":     dict(use_predictive_attention=False, use_latent_topology=True,  use_gradient_routing=False),
}

for task in TASKS:
    benchmark_results[task] = {}
    print(f"\n-- Task: {task} --")
    for bname, flags in baseline_variant_map.items():
        cfg = base_cfg(**flags)
        r = run_variant(task, cfg, SEED, f"/tmp/graft_bench/{task}/{bname}")
        benchmark_results[task][bname] = {
            "final_val_loss": r["final_val_loss"],
            "best_val_loss":  r["best_val_loss"],
            "time_s":         r["time_s"],
        }
        print(f"  {bname:<28} val={r['final_val_loss']:.4f}  best={r['best_val_loss']:.4f}  {r['time_s']:.1f}s")

Path("outputs/benchmark_results.json").write_text(json.dumps(benchmark_results, indent=2))
print("\nSaved outputs/benchmark_results.json")

# ── Parameter counts ─────────────────────────────────────────────────────────
print("\n====== PARAMETER COUNTS ======")

def count_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Break down by module group
    pa = sum(p.numel() for n, p in model.named_parameters() if "predictive" in n and p.requires_grad)
    lt = sum(p.numel() for n, p in model.named_parameters() if "topology" in n and p.requires_grad)
    ex = sum(p.numel() for n, p in model.named_parameters() if "expert" in n and p.requires_grad)
    return {"total": total, "predictive_attention": pa, "latent_topology": lt, "experts": ex}

param_counts = {}
for label, flags in [
    ("full_model",              dict(use_predictive_attention=True,  use_latent_topology=True,  use_gradient_routing=True)),
    ("no_predictive_attention", dict(use_predictive_attention=False, use_latent_topology=True,  use_gradient_routing=True)),
    ("no_latent_topology",      dict(use_predictive_attention=True,  use_latent_topology=False, use_gradient_routing=True)),
    ("no_gradient_routing",     dict(use_predictive_attention=True,  use_latent_topology=True,  use_gradient_routing=False)),
    ("no_topology_no_routing",  dict(use_predictive_attention=True,  use_latent_topology=False, use_gradient_routing=False)),
]:
    cfg = GraftNetConfig()  # use full-size config for param counts
    cfg = replace(cfg, **flags)
    pdict = count_params(GraftNetBackbone(cfg))
    param_counts[label] = pdict
    print(f"  {label:<32} total={pdict['total']:,}  pa={pdict['predictive_attention']:,}  lt={pdict['latent_topology']:,}  ex={pdict['experts']:,}")

Path("outputs/param_counts.json").write_text(json.dumps(param_counts, indent=2))
print("\nAll results saved to outputs/")
