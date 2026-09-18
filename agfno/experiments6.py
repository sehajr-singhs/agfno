"""Third-architecture test: does the collapse-and-rescue mechanism transfer?

Why this experiment exists
--------------------------
The paper's central claim is not "FNO has a bug" but "the band-limited
periodic global channel -- the defining component of the Fourier-operator
family -- forgets geometry with depth, and a zero-gated geometry channel
restores it". A single architecture pair (FNO/AGF-NO) cannot separate those.
This suite runs the SAME multi-seed depth protocol as ``experiments3.py``
(byte-identical data, identical budget, depths 4/8/16, 3 seeds) on:

* ``afno``     -- Adaptive FNO (Guibas et al., 2022; FourCastNet, Pathak et
                  al. 2022): soft spectrum shrinkage + pre-norm residual
                  design instead of FNO's hard truncation + post-norm. Same
                  mode budget, same parameter count (4.81M, ratio 1.000), a
                  different normalization/skip/spectrum treatment -- the
                  controlled third member of the family.
* ``agfafno``  -- the mechanism transplant: identical AFNO mixer plus the
                  zero-gated SDF modulation + pointwise re-injection of
                  ``AGFSpectralBlock``. At initialization it is *exactly*
                  ``afno`` (both gates are zero; verified by test), so the
                  comparison is controlled in the same way as FNO vs AGF-NO.

Falsifiable predictions (pre-registered as gates A1-A4):
  A1  AFNO collapses at depth 16 like FNO (mean rel-L2 >= 0.9);
  A2  AGF-AFNO depth 16 is its own best depth cell (depth is a resource
      once geometry is re-injected);
  A3  the transplant rescues at least half the collapse (afno16/agfafno16
      mean ratio >= 2);
  A4  the linear geometry probe shows the mechanism, not just the symptom:
      AFNO's final-block geometry R^2 decays with depth MORE than
      AGF-AFNO's does.

Output: ``runs/experiments6/summary.json`` with per-cell mean +/- std, the
gates, and the inline probe curves (final-block R^2 by depth for all four
architecture arms once merged with the FNO/AGF-NO probe results already in
the paper).
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
from .experiments2 import _eval_darcy  # same protocol/keys as exp3's sweep
from .probe import extract_latents, linear_probe_geometry
from .models import build_model
from .train import to_tensors, train_one

SEEDS = (0, 1, 2)
DEPTHS = (4, 8, 16)
VARIANTS = (("afno", "afno"), ("agfafno", "agfafno"))


def _save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_all(artifacts_root: str = "/kaggle/working", quick: bool = False,
            models=None) -> dict:
    """Run the sweep. ``models`` (optional) restricts arms, e.g. ``("afno",)``
    -- the kernel split runs one arm per 12 h session in parallel; cross-arm
    gates (A3/A4) are then recomputed at merge time by ``merge_exp6.py``."""
    variants = (VARIANTS if models is None
                else tuple(v for v in VARIANTS if v[0] in tuple(models)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[experiments6] device = {device} arms={[l for l, _ in variants]}")
    vol_root = (
        C.INFRA.volume_mount
        if os.path.isdir(C.INFRA.volume_mount)
        else "/tmp/agfno_vol"
    )
    os.makedirs(vol_root, exist_ok=True)
    exp_root = os.path.join(artifacts_root, "runs", "experiments6")
    os.makedirs(exp_root, exist_ok=True)

    # ---- protocol: parity with experiments3 (byte-identical data) --------- #
    epochs = 2 if quick else C.EXPERIMENT.depth_epochs
    cfg_train = replace(C.TRAIN, num_epochs=epochs)
    depths = (1, 2) if quick else DEPTHS
    seeds = (0,) if quick else SEEDS
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
        for label, name in variants:
            for seed in seeds:
                tag = f"d{depth}_{label}_s{seed}"
                rd = os.path.join(exp_root, tag)
                t0 = time.time()
                # Resume: skip cells whose run_results.json is valid JSON.
                out_path = os.path.join(rd, "run_results.json")
                if os.path.exists(out_path):
                    try:
                        with open(out_path) as f:
                            rec = json.load(f)
                        assert "rel_l2" in rec
                        runs.setdefault((depth, label), {})[seed] = {
                            k: rec[k] for k in ("rel_l2", "rel_l2_ring", "wall_viol")
                        }
                        print(f"[experiments6] {tag}: RESUMED rel_l2 {rec['rel_l2']:.4f}")
                        continue
                    except Exception:
                        pass  # corrupt -> retrain
                model, _ = train_one(name, ddata, C.MODEL, cfg_train, device, rd,
                                     quick, seed=seed, n_blocks=depth)
                m = _eval_darcy(model, ddata, device)
                runs.setdefault((depth, label), {})[seed] = m
                _save_json(out_path, {"depth": depth, "model": label, "seed": seed, **m})
                print(f"[experiments6] {tag} ({(time.time()-t0)/60:.1f} min): " +
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

    # ---- gates (pre-registered, mechanism-faithful) ------------------------ #
    # Arm-aware: A1 needs only the afno arm, A2 only the agfafno arm; A3/A4
    # need both and are recomputed by merge_exp6.py when the kernels run split.
    gates = {}
    if not quick and f"d16_afno" in agg:
        f16 = agg["d16_afno"]
        gates["A1_afno16_mean_rel_l2"] = f16["rel_l2_mean"]
        gates["A1_afno16_collapsed"] = bool(f16["rel_l2_mean"] >= 0.9)
        gates["A1_afno16_worst_seed_best_case"] = float(min(f16["rel_l2_per_seed"]))
    if not quick and f"d16_agfafno" in agg:
        a16 = agg["d16_agfafno"]
        gates["A2_agfafno16_is_best_agf_cell"] = bool(
            a16["rel_l2_mean"] <= min(
                agg[f"d{d}_agfafno"]["rel_l2_mean"] for d in depths
            ) + 1e-6
        )
        gates["A2_agfafno16_mean_rel_l2"] = a16["rel_l2_mean"]
    if not quick and ("d16_afno" in agg and "d16_agfafno" in agg):
        f16, a16 = agg["d16_afno"], agg["d16_agfafno"]
        gates["A3_transplant_ratio"] = float(f16["rel_l2_mean"] / a16["rel_l2_mean"])
        gates["A3_rescued"] = bool(f16["rel_l2_mean"] / a16["rel_l2_mean"] >= 2.0)
        gates["A3_afno16_vs_agfafno16_ring_ratio"] = float(
            f16["rel_l2_ring_mean"] / a16["rel_l2_ring_mean"])
        gates["ALL_GATES_PASS"] = bool(gates["A1_afno16_collapsed"]
                                       and gates["A2_agfafno16_is_best_agf_cell"]
                                       and gates["A3_rescued"])

    # ---- inline geometry probe on the sweep checkpoints (seed 0) ---------- #
    probe_summary = {}
    if not quick:
        probe_summary = _run_probe(exp_root, ddata, device, depths, variants)
        gates_p = {
            "A4_probe_afno_drop": probe_summary.get("r2_drop", {}).get("afno"),
            "A4_probe_agfafno_drop": probe_summary.get("r2_drop", {}).get("agfafno"),
        }
        if None not in gates_p.values():
            gates_p["A4_mechanism_shown"] = bool(
                gates_p["A4_probe_afno_drop"] > gates_p["A4_probe_agfafno_drop"])
            gates["A4_probe_afno_drop"] = gates_p["A4_probe_afno_drop"]
            gates["A4_probe_agfafno_drop"] = gates_p["A4_probe_agfafno_drop"]
            gates["A4_mechanism_shown"] = gates_p["A4_mechanism_shown"]
    core = [gates[k] for k in ("A1_afno16_collapsed", "A2_agfafno16_is_best_agf_cell",
                               "A3_rescued", "A4_mechanism_shown") if k in gates]
    if core:
        gates["ALL_GATES_PASS"] = bool(all(core))

    summary = {"depth": agg, "gates": gates, "probe": probe_summary,
               "config": {"depths": list(depths), "seeds": list(seeds),
                          "epochs": epochs}}
    _save_json(os.path.join(exp_root, "summary.json"), summary)
    _figure(agg, os.path.join(exp_root, "afno_forgetting_curve.png"))
    print("[experiments6] gates:", json.dumps(gates, indent=2))
    return {"run_dir": exp_root, "summary": summary}


def _run_probe(exp_root: str, ddata: dict, device: str, depths, variants) -> dict:
    """Linear geometry probe on the seed-0 sweep checkpoints (this run's arms)."""
    results: dict[str, dict[int, list]] = {l: {} for l, _ in variants}
    te = ddata["test"]
    n_probe = min(192, te["x"].shape[0])
    x, g = te["x"][:n_probe], te["g"][:n_probe]
    for label, name in variants:
        for depth in depths:
            ckpt = os.path.join(exp_root, f"d{depth}_{label}_s{C.SEED}",
                                f"{name}_best.pt")
            if not os.path.exists(ckpt):
                print(f"[experiments6] probe skip {label} d{depth}: no ckpt")
                continue
            model = build_model(name, replace(C.MODEL, n_blocks=depth)).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            latents, sdf = extract_latents(model, x, g)
            per_depth = []
            for l, h in enumerate(latents):
                r = linear_probe_geometry(h, sdf)
                per_depth.append({"block": l + 1, **r})
            results[label][depth] = per_depth
            print(f"[experiments6] probe {label} d{depth}: final-block "
                  f"R2={per_depth[-1]['r2']:.4f}")
            del model

    final_r2 = {m: {int(d): c[-1]["r2"] for d, c in curve.items()}
                for m, curve in results.items()}
    drop = {}
    for m, pts in final_r2.items():
        ds = sorted(pts)
        if len(ds) >= 2:
            drop[m] = float(pts[ds[0]] - pts[ds[-1]])
    return {"final_block_r2_by_depth": final_r2, "r2_drop_depth4_to_16": drop,
            "per_block_r2": {m: {str(d): v for d, v in c.items()}
                             for m, c in results.items()}}


def _figure(agg: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, title in (
        (axes[0], "rel_l2", "global rel $L^2$"),
        (axes[1], "rel_l2_ring", "near-boundary rel $L^2$"),
    ):
        for label, color, name in (("afno", "tab:purple", "AFNO"),
                                   ("agfafno", "tab:green", "AGF-AFNO")):
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
    fig.suptitle("AFNO arms: geometric forgetting vs depth, mean $\\pm$ std over 3 seeds")
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
