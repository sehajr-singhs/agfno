"""Controlled-experiment suite for AGF-NO: the evidence stack behind the paper.

All runs share one byte-identical dataset (same seed 0 as the headline
research run, so test metrics are directly comparable), and every variant
gets the identical training budget. Differences are therefore attributable
to the architecture / loss variant alone.

Experiment matrix
-----------------
1. Headline pair, 3 seeds each      fno / agfno          -> mean +/- std
2. Capacity/mechanism control       agfno_frozen: both geometry gates frozen
   at 0. The network keeps every AGF-NO parameter but the mechanism is off,
   so it must behave like the FNO baseline. If agfno_frozen tracks fno while
   agfno beats it, the gain comes from the mechanism, not extra parameters.
3. Component ablation               agfno_spec_only: anti-forgetting MLP
   re-injection frozen at 0, spectral gating active -- isolates the
   contribution of the re-injection path.
4. Loss ablation                    fno_nopenalty / agfno_nopenalty: trained
   without the boundary-penalty term -- isolates whether the near-boundary
   gain is architectural or an artifact of the loss term.
5. Geometric-forgetting sweep       fno / agfno at depths 1..3 (plus the
   depth-4 main runs): boundary error vs depth. If the anti-forgetting
   injection works, AGF-NO's boundary error should NOT grow with depth the
   way the plain FNO's does.
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
from .train import to_tensors, train_one

# (label, model_name, train_one kwargs) -- every variant shares the budget.
VARIANTS = [
    ("fno", "fno", dict(use_boundary_penalty=True, gate_mode="full")),
    ("agfno", "agfno", dict(use_boundary_penalty=True, gate_mode="full")),
    ("agfno_frozen", "agfno", dict(use_boundary_penalty=True, gate_mode="frozen")),
    ("agfno_spec_only", "agfno", dict(use_boundary_penalty=True, gate_mode="spec_only")),
    ("fno_nopenalty", "fno", dict(use_boundary_penalty=False, gate_mode="full")),
    ("agfno_nopenalty", "agfno", dict(use_boundary_penalty=False, gate_mode="full")),
]

# Variants that get multiple seeds for error bars.
SEEDED = {"fno", "agfno"}


def _evaluate(model, data: dict, device: str, quick: bool) -> dict:
    """Test-split metrics + (optional) zero-shot 2x super-resolution."""
    te = data["test"]
    m = U.eval_metrics(
        model, te["x"], te["g"], te["u"], te["interior"], te["ring"]
    )
    sr = {}
    if not quick and "test_hi" in data:
        sr = U.super_res_eval(
            model, data["test_hi"]["x"], data["test_hi"]["g"],
            data["test_hi"]["u"], device,
        )
    return {**m, **sr}


def _save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_all(artifacts_root: str = "/kaggle/working", quick: bool = False) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[experiments] device = {device}")
    cfg = C.EXPERIMENT

    vol_root = (
        C.INFRA.volume_mount
        if os.path.ismount(C.INFRA.volume_mount) or os.path.isdir(C.INFRA.volume_mount)
        else "/tmp/agfno_vol"
    )
    os.makedirs(vol_root, exist_ok=True)

    # ---------------- Data (shared, byte-identical across runs) ------------
    n_train = min(cfg.n_train, 256) if quick else cfg.n_train
    n_val = min(cfg.n_val, 64) if quick else cfg.n_val
    n_test = min(cfg.n_test, 64) if quick else cfg.n_test
    splits = {
        "train": D.load_or_make_split("train", n_train, cfg.res, vol_root, device),
        "val": D.load_or_make_split("val", n_val, cfg.res, vol_root, device),
        "test": D.load_or_make_split("test", n_test, cfg.res, vol_root, device),
    }
    if not quick:
        splits["test_hi"] = D.load_or_make_split(
            "test", cfg.n_test_hi, cfg.res * 2, vol_root, device
        )
    train_n, val_n, stats = D.normalize_split(splits["train"], splits["val"])
    _, test_n, _ = D.normalize_split(splits["train"], splits["test"])
    data = {
        "train": to_tensors(train_n, device),
        "val": to_tensors(val_n, device),
        "test": to_tensors(test_n, device),
    }
    if not quick:
        _, test_hi_n, _ = D.normalize_split(splits["train"], splits["test_hi"])
        data["test_hi"] = to_tensors(test_hi_n, device)

    exp_root = os.path.join(artifacts_root, "runs", "experiments")
    os.makedirs(exp_root, exist_ok=True)

    # ---------------- Main matrix ------------------------------------------
    epochs = 3 if quick else cfg.epochs
    cfg_train = replace(C.TRAIN, num_epochs=epochs)
    results: dict[str, dict] = {}
    for label, model_name, tkw in VARIANTS:
        seeds = cfg.seeds if label in SEEDED else (C.SEED,)
        for seed in seeds:
            tag = f"{label}_s{seed}"
            run_dir = os.path.join(exp_root, tag)
            os.makedirs(run_dir, exist_ok=True)
            t0 = time.time()
            model, hist = train_one(
                model_name, data, C.MODEL, cfg_train, device, run_dir,
                quick, seed=seed, **tkw,
            )
            dt = time.time() - t0
            m = _evaluate(model, data, device, quick)
            n_params = sum(p.numel() for p in model.parameters())
            results.setdefault(label, {})[seed] = {
                "metrics": m, "train_min": dt / 60.0, "params": n_params,
            }
            print(
                f"[experiments] {tag}: trained {dt/60:.1f} min | "
                f"rel_l2 {m['rel_l2']:.4f} ring {m['rel_l2_ring']:.4f} "
                f"wall {m['wall_viol']:.4f} sr {m.get('rel_l2_sr', float('nan')):.4f}"
            )
            _save_json(os.path.join(run_dir, "run_results.json"),
                       {"label": label, "seed": seed, "train_min": round(dt / 60, 1),
                        "params": n_params, **m})
            torch.save(model.state_dict(), os.path.join(run_dir, "final.pt"))

    # ---------------- Geometric-forgetting depth sweep ---------------------
    depth_epochs = 3 if quick else cfg.depth_epochs
    cfg_depth = replace(C.TRAIN, num_epochs=depth_epochs)
    depth_results = {}
    for depth in cfg.depths:
        for label, model_name in (("fno", "fno"), ("agfno", "agfno")):
            tag = f"depth{depth}_{label}"
            run_dir = os.path.join(exp_root, tag)
            os.makedirs(run_dir, exist_ok=True)
            t0 = time.time()
            model, _ = train_one(
                model_name, data, C.MODEL, cfg_depth, device, run_dir,
                quick, seed=C.SEED, n_blocks=depth,
            )
            m = _evaluate(model, data, device, quick)
            depth_results[tag] = {"depth": depth, "model": label, **m}
            print(f"[experiments] {tag}: rel_l2 {m['rel_l2']:.4f} "
                  f"ring {m['rel_l2_ring']:.4f} wall {m['wall_viol']:.4f}")
            torch.save(model.state_dict(), os.path.join(run_dir, "final.pt"))

    # ---------------- Aggregation ------------------------------------------
    summary = {}
    for label, per_seed in results.items():
        mets = [v["metrics"] for v in per_seed.values()]
        agg = {"n_seeds": len(mets)}
        for k in mets[0]:
            vals = [m[k] for m in mets]
            agg[f"{k}_mean"] = float(np.mean(vals))
            if len(vals) > 1:
                agg[f"{k}_std"] = float(np.std(vals))
        agg["params"] = results[label][C.SEED]["params"]
        summary[label] = agg
    # depth-4 points for the curve = the main-run seed-0 entries
    for label in ("fno", "agfno"):
        m0 = results[label][C.SEED]["metrics"]
        depth_results[f"depth4_{label}"] = {"depth": 4, "model": label, **m0}

    # Headline deltas with the controls interpreted
    def _g(label, key, stat="mean"):
        return summary[label][f"{key}_{stat}"]

    controls = {
        "control_frozen_vs_fno_rel_l2": _g("agfno_frozen", "rel_l2") / _g("fno", "rel_l2"),
        "control_frozen_vs_fno_ring": _g("agfno_frozen", "rel_l2_ring") / _g("fno", "rel_l2_ring"),
        "mechanism_gain_rel_l2": _g("agfno", "rel_l2") / _g("agfno_frozen", "rel_l2"),
        "mechanism_gain_ring": _g("agfno", "rel_l2_ring") / _g("agfno_frozen", "rel_l2_ring"),
        "mlp_path_share_ring": 1.0 - _g("agfno_spec_only", "rel_l2_ring") / _g("agfno", "rel_l2_ring"),
        "penalty_share_agfno_ring": 1.0 - _g("agfno_nopenalty", "rel_l2_ring") / _g("agfno", "rel_l2_ring"),
    }
    summary["_controls"] = controls

    # ---------------- Figures ----------------------------------------------
    _figure_matrix(summary, os.path.join(exp_root, "ablation_matrix.png"))
    _figure_penalty(results, os.path.join(exp_root, "penalty_ablation.png"))
    _figure_depth(depth_results, os.path.join(exp_root, "forgetting_curve.png"))

    _save_json(os.path.join(exp_root, "summary.json"),
               {"summary": summary, "depth": depth_results,
                "data_config": {k: str(v) for k, v in C.DATA.__dict__.items() if not k.startswith("_")},
                "experiment_config": {k: str(v) for k, v in cfg.__dict__.items() if not k.startswith("_")}})
    print("[experiments] summary:")
    print(json.dumps(summary, indent=2))
    return {"run_dir": exp_root, "summary": summary, "depth": depth_results}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _figure_matrix(summary: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["fno", "agfno_frozen", "agfno_spec_only", "agfno"]
    metrics = ["rel_l2", "rel_l2_ring", "wall_viol"]
    titles = ["Relative L2 (global)", "Relative L2 (near boundary)", "Wall fidelity (inside obstacles)"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, met, title in zip(axes, metrics, titles):
        means, errs = [], []
        for lab in labels:
            means.append(summary[lab][f"{met}_mean"])
            errs.append(summary[lab].get(f"{met}_std", 0.0))
        x = np.arange(len(labels))
        ax.bar(x, means, 0.6, yerr=errs, capsize=4, color=["#888", "#bbb", "#6cf", "#d33"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablation matrix (identical budget; error bars = 3 seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_penalty(results: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [("fno", "fno_nopenalty"), ("agfno", "agfno_nopenalty")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (a, b) in zip(axes, pairs):
        met_a = results[a][0]["metrics"]
        met_b = results[b][0]["metrics"]
        x = np.arange(2)
        ax.bar(x - 0.18, [met_a["rel_l2"], met_a["rel_l2_ring"]], 0.36, label="with penalty", color="#999")
        ax.bar(x + 0.18, [met_b["rel_l2"], met_b["rel_l2_ring"]], 0.36, label="no penalty", color="#d33")
        ax.set_xticks(x)
        ax.set_xticklabels(["rel L2", "rel L2 (ring)"])
        ax.set_title(f"{a} vs {b}", fontsize=10)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("Boundary-penalty ablation: is the near-boundary gain architectural?", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_depth(depth_results: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, met in zip(axes, ["rel_l2", "rel_l2_ring"]):
        for model, color in (("fno", "#888"), ("agfno", "#d33")):
            pts = sorted(
                (v["depth"], v[met]) for v in depth_results.values()
                if v["model"] == model
            )
            ds = [p[0] for p in pts]
            es = [p[1] for p in pts]
            ax.plot(ds, es, "o-", color=color, label=f"{model} baseline" if model == "fno" else "AGF-NO")
        ax.set_xlabel("number of Fourier blocks (depth)")
        ax.set_ylabel(met)
        ax.set_yscale("log")
        ax.set_title(f"{met} vs depth", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Geometric forgetting: does boundary error grow with depth?", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)