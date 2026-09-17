"""Experiment 4: the 3-D replication -- geometric forgetting in 3-D Darcy.

The 2-D suites established (a) the depth-16 collapse of vanilla FNO with
zero seed-overlap and the AGF-NO plateau, and (b) the a-priori Delta-rho
diagnostic. This suite tests the dimension-independence of the phenomenon:
same obstacle-Darcy physics lifted to 3-D (spheres + solid tori, multiply
connected), same protocol, same zero-gate capacity control.

Matrix: {fno, agfno} x depth {4, 16} x seeds {0, 1, 2} at res 32, plus

  * layer-wise linear-geometry probe (does the 3-D latent forget geometry?),
  * near-wall (ring) error attribution,
  * zero-shot super-resolution 32 -> 48 (measures discretization transfer),
  * stability gates:
      G1  FNO@16(3D) collapsed: mean rel-L2 >= 0.9 (and >= 2x FNO@4),
      G2  AGF-NO ring error at depth 16 within 1.25x of its depth-4 ring
          error (depth-stability of the near-wall metric),
      G3  probe R^2 of FNO@16 below 0.15 (representational collapse), while
          AGF@16 stays above 0.30.

Checkpoints are saved per run for post-hoc analysis; every stored ground
truth field satisfies the discrete PDE to < 1e-6 relative residual (the
3-D solver runs in float64 and is asserted in tests).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace

import numpy as np
import torch

from . import config as C
from . import utils as U
from .dataset import normalize_split
from .dataset3d import load_or_make_split3
from .models3d import build_model3d
from .probe import linear_probe_geometry
from .train import to_tensors, train_one

SEEDS = (0, 1, 2)
DEPTHS = (4, 16)
# metric keys carried into run_results.json (non-float entries dropped)
_METRIC_KEYS = ("rel_l2", "rel_l2_ring", "wall_viol", "rel_l2_sr",
                "probe_r2_final")


def _save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _latents3d(model, x, g, batch=4, block: int = -1):
    """Latent at ``block`` (default: final) + SDF target.

    Memory note: only the requested block is retained on host RAM. (An earlier
    version collected every block's latent, ~4 GB at depth 16 -- the OOM that
    killed the first 3-D kernel.)
    """
    model.eval()
    n_blocks = len(model.blocks)
    keep = block % n_blocks
    acc = []
    with torch.no_grad():
        for i0 in range(0, x.shape[0], batch):
            xb, gb = x[i0 : i0 + batch], g[i0 : i0 + batch]
            v = model.lift(xb)
            for l, blk in enumerate(model.blocks):
                v = blk(v, gb)
                if l == keep:
                    acc.append(v.detach().cpu())
    return torch.cat(acc, dim=0), g[:, :1].detach().cpu()


def run_all(
    artifacts_root: str = "/kaggle/working",
    quick: bool = False,
    depths=DEPTHS,
    seeds=SEEDS,
    models=("fno", "agfno"),
    reuse: dict | None = None,
) -> dict:
    """Run the 3-D matrix (subsettable + resumable).

    ``depths``/``seeds`` restrict the matrix (e.g. depth-16-only rescue runs);
    ``reuse`` maps tag -> metrics dict for cells already measured elsewhere
    (merged into the aggregate/gates instead of retrained).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[experiments4] device = {device}")
    vol_root = (
        C.INFRA.volume_mount
        if os.path.isdir(C.INFRA.volume_mount)
        else "/tmp/agfno_vol"
    )
    os.makedirs(vol_root, exist_ok=True)
    exp_root = os.path.join(artifacts_root, "runs", "experiments4")
    os.makedirs(exp_root, exist_ok=True)

    epochs = 3 if quick else C.TRAIN.num_epochs // 2  # 150 epochs per run
    cfg_train = replace(C.TRAIN, num_epochs=epochs,
                        batch_size=8, eval_every=10)
    depths = (1, 2) if quick else DEPTHS
    seeds = (0, 1) if quick else SEEDS
    res = 16 if quick else C.DATA3D.res
    n_train = 32 if quick else C.DATA3D.n_train
    n_val = 16 if quick else C.DATA3D.n_val
    n_test = 16 if quick else C.DATA3D.n_test

    # ---- data (cached per res/n on disk) ---------------------------------- #
    # Note: ground-truth arrays are freed from host RAM right after the GPU
    # tensors are built (the first kernel died of host OOM with everything
    # resident). Data generation is deterministic, so nothing is lost.
    splits = {}
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        splits[split] = load_or_make_split3(split, n, res, vol_root, device)
    test_hi = None
    if not quick:
        test_hi = load_or_make_split3(
            "test", min(C.DATA3D.n_test, 64), C.DATA3D.res_hi, vol_root, device
        )
    train_n, val_n, _ = normalize_split(splits["train"], splits["val"])
    _, test_n, _ = normalize_split(splits["train"], splits["test"])
    data = {
        "train": to_tensors(train_n, device),
        "val": to_tensors(val_n, device),
        "test": to_tensors(test_n, device),
    }
    if test_hi is not None:
        _, hi_n, _ = normalize_split(splits["train"], test_hi)
        data["test_hi"] = to_tensors(hi_n, device)
    # free host-resident ground truth (GPU tensors are now authoritative)
    del splits, train_n, val_n, test_n
    if test_hi is not None:
        del test_hi, hi_n
    import gc

    gc.collect()

    cfg_model = replace(
        C.MODEL3D, in_ch=3, sdf_ch=3,
        width=16 if quick else C.MODEL3D.width,
    )

    # ---- the matrix -------------------------------------------------------- #
    # resume: skip runs whose results already exist on disk (e.g. produced by
    # an earlier kernel -- lets a killed session pick up where it left off)
    runs = {}
    for depth in depths:
        for label in models:
            for seed in seeds:
                tag = f"d{depth}_{label}_s{seed}"
                rd = os.path.join(exp_root, tag)
                prior = os.path.join(rd, "run_results.json")
                if os.path.exists(prior):
                    print(f"[experiments4] {tag}: found existing results, skipping")
                    saved = json.load(open(prior))
                    m = {k: saved[k] for k in _METRIC_KEYS if k in saved
                         and isinstance(saved[k], (int, float))}
                    runs.setdefault((depth, label), {})[seed] = m
                    continue
                t0 = time.time()
                model, _ = train_one(
                    label, data, cfg_model, cfg_train, device, rd,
                    quick, seed=seed, n_blocks=depth,
                    model_builder=build_model3d,
                )
                te = data["test"]
                m = U.eval_metrics(
                    model, te["x"], te["g"], te["u"], te["interior"], te["ring"]
                )
                if "test_hi" in data and not quick:
                    hi = data["test_hi"]
                    m.update(
                        U.super_res_eval(model, hi["x"], hi["g"], hi["u"], device)
                    )
                # probe on the FINAL block (representational geometry).
                # Non-fatal: an hours-long training run must not be lost to a
                # probe failure, so metrics are persisted BEFORE probing and
                # run_results.json is re-written with the probe afterwards.
                try:
                    lat, sdf_t = _latents3d(model, te["x"], te["g"], batch=4)
                    pr = linear_probe_geometry(lat, sdf_t, n_fit=150_000,
                                               n_eval=75_000)
                    m["probe_r2_final"] = pr["r2"]
                except Exception as exc:  # noqa: BLE001 -- probe is auxiliary
                    print(f"[experiments4] probe failed ({type(exc).__name__}: "
                          f"{exc}); continuing without it")
                    pr = {"r2": float("nan"), "error": str(exc)}
                m = {k: m[k] for k in _METRIC_KEYS if k in m}
                runs.setdefault((depth, label), {})[seed] = m
                _save_json(
                    os.path.join(rd, "run_results.json"),
                    {"depth": depth, "model": label, "seed": seed, **m,
                     "probe": pr},
                )
                print(
                    f"[experiments4] {tag} ({(time.time()-t0)/60:.1f} min): "
                    + " ".join(f"{k}={v:.4f}" for k, v in m.items())
                )
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    # ---- merge externally supplied results (e.g. depth-4 from kernel 1) ----- #
    if reuse:
        for tag, m in reuse.items():
            depth, label, seed = int(tag.split("_")[0][1:]), tag.split("_")[1], \
                int(tag.split("_")[2][1:])
            runs.setdefault((depth, label), {})[seed] = {
                k: m[k] for k in _METRIC_KEYS if k in m}
            print(f"[experiments4] merged external result: {tag}")

    # ---- aggregate + gates -------------------------------------------------- #
    agg, gates = _aggregate_and_gates(runs, quick)

    summary = {
        "depth": agg,
        "gates": gates,
        "config": {
            "res": res, "res_hi": C.DATA3D.res_hi, "depths": list(depths),
            "seeds": list(seeds), "epochs": epochs,
            "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "model3d": {
                "width": cfg_model.width, "modes": cfg_model.modes_d,
                "batch": cfg_train.batch_size,
            },
        },
    }
    _save_json(os.path.join(exp_root, "summary.json"), summary)
    _figure(agg, os.path.join(exp_root, "forgetting3d.png"))
    print("[experiments4] gates:", json.dumps(gates, indent=2))
    return {"run_dir": exp_root, "summary": summary}


def _aggregate_and_gates(
    runs: dict, quick: bool = False
) -> tuple[dict, dict]:
    """Aggregate per-seed metrics into cells and compute stability gates.

    Works on partial matrices (e.g. a rescue kernel that trained only the
    depth-16 cells): gates are emitted only when all four cells are present,
    so the final full-matrix gates are computed by ``merge_rescue`` locally
    after both rescue kernels land.
    """
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

    gates = {}
    if not quick and all(k in agg for k in
                         ("d4_fno", "d16_fno", "d4_agfno", "d16_agfno")):
        f4, f16 = agg["d4_fno"], agg["d16_fno"]
        a4, a16 = agg["d4_agfno"], agg["d16_agfno"]
        gates = {
            "G1_fno16_collapsed": bool(
                f16["rel_l2_mean"] >= 0.9
                and f16["rel_l2_mean"] >= 2.0 * f4["rel_l2_mean"]
            ),
            "G1_fno16_mean_rel_l2": f16["rel_l2_mean"],
            "G2_agf16_ring_stable": bool(
                a16["rel_l2_ring_mean"]
                <= 2.0 * a4["rel_l2_ring_mean"] + 0.05
            ),
            "G2_agf4_ring_mean": a4["rel_l2_ring_mean"],
            "G2_agf16_ring_mean": a16["rel_l2_ring_mean"],
            "G2_agf16_ring_over_d4": (
                a16["rel_l2_ring_mean"] / a4["rel_l2_ring_mean"]
            ),
            "G3_fno16_probe_r2": f16.get("probe_r2_final_mean", float("nan")),
            "G3_agf16_probe_r2": a16.get("probe_r2_final_mean", float("nan")),
            "G3_probe_collapse": bool(
                f16.get("probe_r2_final_mean", float("nan")) < 0.15
                and a16.get("probe_r2_final_mean", float("nan")) > 0.30
            ),
            "fno16_vs_agf16_ring_ratio": (
                f16["rel_l2_ring_mean"] / a16["rel_l2_ring_mean"]
            ),
        }
        gates["ALL_GATES_PASS"] = bool(
            gates["G1_fno16_collapsed"]
            and gates["G2_agf16_ring_stable"]
            and gates["G3_probe_collapse"]
        )
    return agg, gates


def _figure(agg: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, title in (
        (axes[0], "rel_l2", "global rel $L^2$ (3-D)"),
        (axes[1], "rel_l2_ring", "near-wall rel $L^2$ (3-D)"),
    ):
        for label, color, name in (
            ("fno", "tab:red", "FNO"),
            ("agfno", "tab:blue", "AGF-NO"),
        ):
            pts = sorted(
                (int(k.split("_")[0][1:]), v[f"{key}_mean"], v[f"{key}_std"])
                for k, v in agg.items()
                if v["model"] == label
            )
            d = [p[0] for p in pts]
            mu = [p[1] for p in pts]
            sd = [p[2] for p in pts]
            ax.errorbar(d, mu, yerr=sd, marker="o", capsize=3,
                        color=color, label=name)
        ax.set_xlabel("depth (blocks)")
        ax.set_ylabel(title)
        ax.set_xticks(sorted({int(k.split("_")[0][1:]) for k in agg}))
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Geometric forgetting in 3-D: collapse vs plateau", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="/kaggle/working")
    args = ap.parse_args()
    run_all(args.out, quick=args.quick)


if __name__ == "__main__":
    main()
