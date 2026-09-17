"""Multi-seed depth sweep: error bars for the collapse/plateau result.

The single-seed depth sweep (experiments2) produced the paper's headline
finding -- vanilla FNO collapses at depth 16 while AGF-NO improves with
depth -- but a collapse this dramatic demands error bars before it goes in
a paper. This suite re-runs the sweep with 3 seeds per (depth, model) cell
and reports mean +- std, plus two stability gates:

  G1  FNO@16 must remain collapsed with mean rel-L2 >= 0.9 (mean predictor
      scores ~1.0), i.e. no seed "escapes" the collapse;
  G2  AGF-NO@16 must remain the best AGF cell (mean), i.e. the
      depth-is-a-resource claim is not a seed artifact.

Checkpoints are saved per run so the probe (probe.py) and the frequency
analysis (analysis.py) can be re-run on any cell afterwards.

Every run shares byte-identical data with experiments.py / experiments2.py
(same seed -> same split) and an identical training budget; only the model,
depth, and seed vary.
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
from .experiments2 import _eval_darcy  # same protocol/keys as the earlier sweeps
from .train import to_tensors, train_one

SEEDS = (0, 1, 2)
DEPTHS = (4, 8, 16)


def _save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_all(artifacts_root: str = "/kaggle/working", quick: bool = False) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[experiments3] device = {device}")
    vol_root = (
        C.INFRA.volume_mount
        if os.path.isdir(C.INFRA.volume_mount)
        else "/tmp/agfno_vol"
    )
    os.makedirs(vol_root, exist_ok=True)
    exp_root = os.path.join(artifacts_root, "runs", "experiments3")
    os.makedirs(exp_root, exist_ok=True)

    # ---- data: byte-identical split to the earlier suites ---------------- #
    epochs = 3 if quick else C.EXPERIMENT.depth_epochs
    cfg_train = replace(C.TRAIN, num_epochs=epochs)
    depths = (1, 2) if quick else DEPTHS
    seeds = (0, 1) if quick else SEEDS
    n_train = min(C.EXPERIMENT.n_train, 128) if quick else C.EXPERIMENT.n_train
    n_val = min(C.EXPERIMENT.n_val, 48) if quick else C.EXPERIMENT.n_val
    n_test = min(C.EXPERIMENT.n_test, 96) if quick else C.EXPERIMENT.n_test
    splits = {
        "train": D.load_or_make_split("train", n_train, C.EXPERIMENT.res, vol_root, device),
        "val": D.load_or_make_split("val", n_val, C.EXPERIMENT.res, vol_root, device),
        "test": D.load_or_make_split("test", n_test, C.EXPERIMENT.res, vol_root, device),
    }
    train_n, val_n, _ = D.normalize_split(splits["train"], splits["val"])
    _, test_n, _ = D.normalize_split(splits["train"], splits["test"])
    ddata = {
        "train": to_tensors(train_n, device),
        "val": to_tensors(val_n, device),
        "test": to_tensors(test_n, device),
    }

    # ---- the sweep -------------------------------------------------------- #
    runs = {}  # (depth, label) -> {seed: metrics}
    for depth in depths:
        for label, name in (("fno", "fno"), ("agfno", "agfno")):
            for seed in seeds:
                tag = f"d{depth}_{label}_s{seed}"
                rd = os.path.join(exp_root, tag)
                t0 = time.time()
                model, _ = train_one(name, ddata, C.MODEL, cfg_train, device, rd,
                                     quick, seed=seed, n_blocks=depth)
                m = _eval_darcy(model, ddata, device)
                runs.setdefault((depth, label), {})[seed] = m
                _save_json(os.path.join(rd, "run_results.json"),
                           {"depth": depth, "model": label, "seed": seed, **m})
                print(f"[experiments3] {tag} ({(time.time()-t0)/60:.1f} min): " +
                      " ".join(f"{k}={v:.4f}" for k, v in m.items()))
                del model

    # ---- aggregate: mean/std per cell ------------------------------------- #
    agg = {}
    for (depth, label), per_seed in sorted(runs.items()):
        mets = list(per_seed.values())
        cell = {"n_seeds": len(mets), "depth": depth, "model": label}
        for k in mets[0]:
            vals = [m[k] for m in mets]
            cell[f"{k}_mean"] = float(np.nanmean(vals))
            cell[f"{k}_std"] = float(np.nanstd(vals))
            if len(mets) > 1:
                cell[f"{k}_per_seed"] = [float(v) for v in vals]
        agg[f"d{depth}_{label}"] = cell

    # ---- stability gates --------------------------------------------------- #
    gates = {}
    if not quick:
        f16 = agg["d16_fno"]
        a16 = agg["d16_agfno"]
        a4 = agg["d4_agfno"]
        gates = {
            # G1: FNO depth-16 stays collapsed across seeds
            "G1_fno16_mean_rel_l2": f16["rel_l2_mean"],
            "G1_fno16_collapsed": bool(f16["rel_l2_mean"] >= 0.9),
            "G1_fno16_worst_seed_best_case": float(min(f16["rel_l2_per_seed"])),
            # G2: AGF depth-16 remains its own best depth cell
            "G2_agf16_is_best_agf_cell": bool(
                a16["rel_l2_mean"] <= min(
                    agg[f"d{d}_agfno"]["rel_l2_mean"] for d in depths
                ) + 1e-6
            ),
            "G2_agf16_mean_rel_l2": a16["rel_l2_mean"],
            "G2_agf4_mean_rel_l2": a4["rel_l2_mean"],
            # effect sizes with error bars
            "fno16_vs_agf16_mean_ratio": f16["rel_l2_mean"] / a16["rel_l2_mean"],
            "fno16_std_over_mean": f16["rel_l2_std"] / f16["rel_l2_mean"],
            "agf16_std_over_mean": a16["rel_l2_std"] / a16["rel_l2_mean"],
            "ring_fno16_vs_agf16_ratio": f16["rel_l2_ring_mean"] / a16["rel_l2_ring_mean"],
        }
        gates["ALL_GATES_PASS"] = bool(gates["G1_fno16_collapsed"] and gates["G2_agf16_is_best_agf_cell"])

    summary = {"depth": agg, "gates": gates,
               "config": {"depths": list(depths), "seeds": list(seeds),
                          "epochs": epochs}}
    _save_json(os.path.join(exp_root, "summary.json"), summary)
    _figure(agg, os.path.join(exp_root, "forgetting_curve_seeds.png"))
    print("[experiments3] gates:", json.dumps(gates, indent=2))
    return {"run_dir": exp_root, "summary": summary}


def _figure(agg: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, title in (
        (axes[0], "rel_l2", "global rel $L^2$"),
        (axes[1], "rel_l2_ring", "near-boundary rel $L^2$"),
    ):
        for label, color, name in (("fno", "tab:red", "FNO"),
                                   ("agfno", "tab:blue", "AGF-NO")):
            pts = sorted(
                (int(k.split("_")[0][1:]), v[f"{key}_mean"], v[f"{key}_std"])
                for k, v in agg.items() if v["model"] == label
            )
            d = [p[0] for p in pts]
            mu = [p[1] for p in pts]
            sd = [p[2] for p in pts]
            ax.errorbar(d, mu, yerr=sd, fmt="o-", color=color, capsize=4,
                        label=name, lw=1.8, markersize=5)
        ax.set_xlabel("depth (blocks)")
        ax.set_ylabel(title)
        ax.set_yscale("log")
        ax.set_xticks(sorted({int(k.split("_")[0][1:]) for k in agg}))
        ax.legend()
    fig.suptitle("Geometric forgetting vs depth, mean $\\pm$ std over 3 seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/working")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run_all(args.root, args.quick)
