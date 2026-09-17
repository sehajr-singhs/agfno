"""Canonical Darcy benchmark (piecewise-constant K), compared with the literature.

Protocol (the FNO paper's benchmark family, Li et al. 2021): the unit square
is split into a 4x4 grid of cells, each containing a random circle;
permeability K = 12 inside any circle and 1 outside; the solution of
-div(K grad u) = 1 with u = 0 on the outer frame is generated with our
batched PCG solver (flow passes through the circles -- no interior walls).

Models trained WITHOUT the boundary penalty: this benchmark has no interior
wall conditions, so any near-boundary gain is purely architectural (the SDF
channels + geometry injection), not a loss-term effect.

Published references (85x85 grid, 10k training samples, as reported in the
literature, cited for context -- NOT produced by our pipeline):
    FNO      0.0082   Li et al., ICLR 2021
    Geo-FNO  0.0068   Li et al., JMLR 2023

Our run uses a 64x64 grid with 2000 training samples, so absolute numbers
are not directly comparable to the references; the table reports both with
this caveat. The point is directional: does geometry injection help on the
standard benchmark too, with an FNO / gates-frozen control to attribute it?
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace

import numpy as np
import torch

from . import config as C
from . import dataset as D
from . import utils as U
from .experiments import _evaluate, _save_json
from .train import to_tensors, train_one


def main(artifacts_root: str = "/kaggle/working", quick: bool = False) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[benchmark] device = {device}")
    cfg = C.BENCHMARK

    vol_root = (
        C.INFRA.volume_mount
        if os.path.ismount(C.INFRA.volume_mount) or os.path.isdir(C.INFRA.volume_mount)
        else "/tmp/agfno_vol"
    )
    os.makedirs(vol_root, exist_ok=True)

    res = 16 if quick else cfg.res
    n_train = min(cfg.n_train, 96) if quick else cfg.n_train
    n_val = min(cfg.n_val, 32) if quick else cfg.n_val
    n_test = min(cfg.n_test, 48) if quick else cfg.n_test

    # ---------------- Data (canonical piecewise-constant K) ----------------
    splits = {
        "train": D.load_or_make_split("train", n_train, res, vol_root, device, kind="piecewise"),
        "val": D.load_or_make_split("val", n_val, res, vol_root, device, kind="piecewise"),
        "test": D.load_or_make_split("test", n_test, res, vol_root, device, kind="piecewise"),
    }
    train_n, val_n, stats = D.normalize_split(splits["train"], splits["val"])
    _, test_n, _ = D.normalize_split(splits["train"], splits["test"])
    data = {
        "train": to_tensors(train_n, device),
        "val": to_tensors(val_n, device),
        "test": to_tensors(test_n, device),
    }

    run_dir = os.path.join(artifacts_root, "runs", "benchmark")
    os.makedirs(run_dir, exist_ok=True)

    epochs = 3 if quick else cfg.epochs
    cfg_train = replace(C.TRAIN, num_epochs=epochs)
    # No wall conditions in this benchmark -> the penalty term is meaningless;
    # use_boundary_penalty=False isolates the pure architectural effect.
    variants = [
        ("fno", "fno", dict(use_boundary_penalty=False, gate_mode="full")),
        ("agfno", "agfno", dict(use_boundary_penalty=False, gate_mode="full")),
        ("agfno_frozen", "agfno", dict(use_boundary_penalty=False, gate_mode="frozen")),
    ]

    results = {}
    models = {}
    for label, model_name, tkw in variants:
        tag = f"{label}_s{C.SEED}"
        rd = os.path.join(run_dir, tag)
        os.makedirs(rd, exist_ok=True)
        t0 = time.time()
        model, _ = train_one(
            model_name, data, C.MODEL, cfg_train, device, rd, quick,
            seed=C.SEED, **tkw,
        )
        dt = time.time() - t0
        m = _evaluate(model, data, device, quick)
        results[label] = {**m, "train_min": round(dt / 60, 1)}
        models[label] = model
        print(f"[benchmark] {label}: rel_l2 {m['rel_l2']:.4f} "
              f"ring {m['rel_l2_ring']:.4f} | {dt/60:.1f} min")
        _save_json(os.path.join(rd, "run_results.json"), results[label])
        torch.save(model.state_dict(), os.path.join(rd, "final.pt"))

    # ---------------- Comparison with the literature -----------------------
    comparison = {
        "ours": {
            "FNO (ours, 64x64, 2k train)": float(results["fno"]["rel_l2"]),
            "AGF-NO (ours, 64x64, 2k train)": float(results["agfno"]["rel_l2"]),
            "AGF-NO gates-frozen control": float(results["agfno_frozen"]["rel_l2"]),
            "AGF-NO rel L2 near circle boundaries": float(results["agfno"]["rel_l2_ring"]),
            "FNO rel L2 near circle boundaries": float(results["fno"]["rel_l2_ring"]),
        },
        "published_references": {
            "FNO (Li et al., ICLR 2021, 85x85, 10k train)": cfg.published_fno_rel_l2,
            "Geo-FNO (Li et al., JMLR 2023, 85x85, 10k train)": cfg.published_geofno_rel_l2,
        },
        "caveat": (
            "Published numbers are at 85x85 with 10k training samples; our run "
            "uses 64x64 with 2k samples, so absolute errors are not directly "
            "comparable. The comparison is directional: architecture gains on "
            "the same benchmark family."
        ),
    }

    _figure_benchmark(results, comparison, os.path.join(run_dir, "benchmark_comparison.png"))

    summary = {
        "results": results,
        "comparison": comparison,
        "normalization": {k: float(v) for k, v in stats.items()},
        "data_config": {k: str(v) for k, v in C.DATA.__dict__.items() if not k.startswith("_")},
        "benchmark_config": {k: str(v) for k, v in cfg.__dict__.items() if not k.startswith("_")},
    }
    _save_json(os.path.join(run_dir, "summary.json"), summary)
    print("[benchmark] summary:")
    print(json.dumps(summary, indent=2))
    return {"run_dir": run_dir, "summary": summary}


def _figure_benchmark(results: dict, comparison: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["fno", "agfno_frozen", "agfno"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # Panel 1: our models, global vs near-boundary
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, [results[k]["rel_l2"] for k in labels], 0.36,
                label="rel L2 (global)", color="#888")
    axes[0].bar(x + 0.18, [results[k]["rel_l2_ring"] for k in labels], 0.36,
                label="rel L2 (near circle boundaries)", color="#d33")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, fontsize=9)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Ours (64x64, 2k train, no boundary penalty)")
    axes[0].grid(axis="y", alpha=0.3)
    # Panel 2: ours vs published references
    ours = comparison["ours"]
    refs = comparison["published_references"]
    x2 = np.arange(2)
    axes[1].bar(x2, [ours["AGF-NO (ours, 64x64, 2k train)"], ours["FNO (ours, 64x64, 2k train)"]],
                0.5, color=["#d33", "#888"])
    axes[1].axhline(refs["FNO (Li et al., ICLR 2021, 85x85, 10k train)"], color="#36c",
                    ls="--", lw=1.2, label="FNO (lit., 0.0082)")
    axes[1].axhline(refs["Geo-FNO (Li et al., JMLR 2023, 85x85, 10k train)"], color="#0a0",
                    ls="--", lw=1.2, label="Geo-FNO (lit., 0.0068)")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["AGF-NO", "FNO"], fontsize=10)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].set_title("vs published references (85x85, 10k train)\nnot directly comparable: resolution/sample budget differ")
    fig.suptitle("Canonical Darcy benchmark (piecewise-constant K)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)