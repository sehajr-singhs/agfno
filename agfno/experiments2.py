"""Extended evidence suite: second PDE family, deeper forgetting sweep, Geo-FNO baseline.

Closes the three remaining external-validity gaps of the AGF-NO evidence stack:

A. Second PDE family -- advection-diffusion past fixed-temperature obstacles
   (the SAME irregular polygonal geometry as Darcy, but *dynamical*):
   sharp thermal boundary layers coupled to geometry, plus a NEW generalization
   axis: zero-shot rollout in time (the operator is composed with itself to
   reach 2T, twice its training horizon).

B. Extended geometric-forgetting sweep -- depths 4/6/8/16 (the earlier suite
   covered 1-4). If forgetting worsens with depth, AGF-NO's advantage must
   GROW with depth; the linear probe (probe.py) quantifies the mechanism.

C. Geo-FNO-style baseline (learned deformation + standard FNO) on the obstacle
   Darcy benchmark. Structural note: Geo-FNO's deformation is a diffeomorphism,
   but a domain with interior wall obstacles is NOT simply connected -- no
   deformation can flatten it. DeformFNO (identity at init, unit-tested) is the
   honest strongest-effort representative of the deformation family here.

Every run shares byte-identical data with experiments.py (same seeds) and an
identical training budget per variant; differences are attributable to the
architecture alone.
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

# PDE2 variants: headline pair + frozen capacity control, 3 seeds.
PDE2_VARIANTS = [
    ("fno", "fno", dict(use_boundary_penalty=True, gate_mode="full")),
    ("agfno", "agfno", dict(use_boundary_penalty=True, gate_mode="full")),
    ("agfno_frozen", "agfno", dict(use_boundary_penalty=True, gate_mode="frozen")),
]
PDE2_SEEDED = {"fno", "agfno"}
# Extended depth sweep (4 is re-run here so the curve is self-contained).
DEPTHS = (4, 6, 8, 16)


# --------------------------------------------------------------------------- #
# PDE2 data plumbing
# --------------------------------------------------------------------------- #
def to_advec_tensors(d: dict, device: str) -> dict:
    """Advection-split arrays -> the tensor dict train_one expects.

    Model input  x = [u0, sdf, ring, vx, vy]   (in_ch = 5)
    Geometry     g = [sdf, ring, interior]     (sdf_ch = 3, unchanged)
    Targets      u  = normalized u_T           (training horizon)
                 u_long = normalized u_long    (2T, zero-shot rollout target)

    u_T and u_long are normalized with the SAME train-split stats, so the
    learned operator (which maps normalized->normalized) can be composed with
    itself for the rollout test.
    """
    t = {}
    for k in ("a", "u_T", "u_long", "v", "sdf", "interior", "ring"):
        t[k] = torch.tensor(np.ascontiguousarray(d[k]), dtype=torch.float32, device=device)
    t["u"] = t["u_T"]  # train_one's training-target key (normalized u at T)
    t["x"] = torch.cat([t["a"], t["sdf"], t["ring"], t["v"]], dim=1)
    t["g"] = torch.cat([t["sdf"], t["ring"], t["interior"]], dim=1)
    return t


def _load_advec(root: str, n: int, res: int, device: str) -> dict:
    path = os.path.join(root, C.ADVEC.shard_name("all", n, res))
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    import hashlib

    off = int(hashlib.md5(b"advec-all").hexdigest(), 16) % (2**31)
    d = D.make_advec_split("all", n, res, device)
    os.makedirs(root, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in d.items() if k != "meta"})
    return d


def advec_data(vol_root: str, device: str, quick: bool) -> tuple[dict, dict]:
    """Build normalized PDE2 tensors + the (u_mu, u_sd) stats used."""
    cfg = C.ADVEC
    n_train = min(cfg.n_train, 96) if quick else cfg.n_train
    n_val = min(cfg.n_val, 48) if quick else cfg.n_val
    n_test = min(cfg.n_test, 48) if quick else cfg.n_test
    raw = _load_advec(vol_root, n_train + n_val + n_test, cfg.res, device)
    n = n_train + n_val + n_test
    sl = slice(0, n)
    a_mu, a_sd = raw["a"][:n].mean(), raw["a"][:n].std() + 1e-8
    u_mu, u_sd = raw["u_T"][:n].mean(), raw["u_T"][:n].std() + 1e-8
    d = {k: v for k, v in dict(raw).items() if k != "meta"}
    d["a"] = (d["a"][:n] - a_mu) / a_sd
    d["u_T"] = (d["u_T"][:n] - u_mu) / u_sd
    d["u_long"] = (d["u_long"][:n] - u_mu) / u_sd
    tr = {k: d[k][0:n_train] for k in d}
    va = {k: d[k][n_train : n_train + n_val] for k in d}
    te = {k: d[k][n_train + n_val : n] for k in d}
    data = {
        "train": to_advec_tensors(tr, device),
        "val": to_advec_tensors(va, device),
        "test": to_advec_tensors(te, device),
    }
    stats = dict(a_mu=float(a_mu), a_sd=float(a_sd), u_mu=float(u_mu), u_sd=float(u_sd))
    return data, stats


# --------------------------------------------------------------------------- #
# PDE2 evaluation: one-step (T) + zero-shot composed rollout (2T)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_pde2(model, data: dict, device: str) -> dict:
    """One-step accuracy at T plus zero-shot generalization to 2T.

    Rollout: the operator G maps (u(0), v, geom) -> u(T) in normalized space.
    Composing, u(2T) ~= G(G(u(0))): feed the T-horizon prediction back in as
    channel 0, keeping geometry and velocity channels fixed.
    """
    model.eval()
    te = data["test"]
    m = U.eval_metrics(model, te["x"], te["g"], te["u"], te["interior"], te["ring"])

    outs_2T = []
    bs = 64
    for i0 in range(0, te["x"].shape[0], bs):
        x = te["x"][i0 : i0 + bs]
        g = te["g"][i0 : i0 + bs]
        y2 = te["u_long"][i0 : i0 + bs]
        pT = model(x, g)
        x2 = x.clone()
        x2[:, 0:1] = pT  # feed prediction back in; geometry + velocity kept
        p2 = model(x2, g)
        outs_2T.append(p2)
    p2 = torch.cat(outs_2T)
    y2 = te["u_long"]
    err = ((p2 - y2) ** 2).mean(dim=(1, 2, 3))
    den = (y2**2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    rel2T = float(torch.sqrt(err / den).mean())
    # boundary-resolved error at 2T (inside obstacle ring cells)
    ring = te["ring"].bool()
    errb = ((p2 - y2) ** 2)[ring].sum()
    denb = (y2**2)[ring].sum().clamp_min(1e-12)
    rel2T_ring = float(torch.sqrt(errb / denb))
    out = dict(rel_l2_T=m["rel_l2"], rel_l2_ring_T=m["rel_l2_ring"],
               wall_viol_T=m["wall_viol"], rel_l2_2T=rel2T,
               rel_l2_ring_2T=rel2T_ring,
               rollout_ratio=rel2T / max(m["rel_l2"], 1e-12))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_all(artifacts_root: str = "/kaggle/working", quick: bool = False) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[experiments2] device = {device}")
    vol_root = (
        C.INFRA.volume_mount
        if os.path.isdir(C.INFRA.volume_mount)
        else "/tmp/agfno_vol"
    )
    os.makedirs(vol_root, exist_ok=True)
    exp_root = os.path.join(artifacts_root, "runs", "experiments2")
    os.makedirs(exp_root, exist_ok=True)
    summary: dict = {}

    # ================= A. Second PDE family (advection-diffusion) ========== #
    epochs = 3 if quick else 120
    cfg_train = replace(C.TRAIN, num_epochs=epochs)
    cfg_model = replace(C.MODEL, in_ch=5)  # [a, sdf, ring, vx, vy]
    adata, astats = advec_data(vol_root, device, quick)
    pde2: dict = {}
    for label, name, tkw in PDE2_VARIANTS:
        seeds = C.EXPERIMENT.seeds if (label in PDE2_SEEDED and not quick) else (C.SEED,)
        for seed in seeds:
            tag = f"pde2_{label}_s{seed}"
            rd = os.path.join(exp_root, tag)
            t0 = time.time()
            model, _ = train_one(name, adata, cfg_model, cfg_train, device, rd,
                                 quick, seed=seed, **tkw)
            m = eval_pde2(model, adata, device)
            pde2.setdefault(label, {})[seed] = m
            print(f"[experiments2] {tag} ({(time.time()-t0)/60:.1f} min): " +
                  " ".join(f"{k}={v:.4f}" for k, v in m.items()))
            _save_json(os.path.join(rd, "run_results.json"),
                       {"label": label, "seed": seed, **m})
    # aggregate
    p2agg = {}
    for label, per_seed in pde2.items():
        mets = list(per_seed.values())
        agg = {"n_seeds": len(mets)}
        for k in mets[0]:
            vals = [m[k] for m in mets]
            agg[f"{k}_mean"] = float(np.nanmean(vals))
            if len(vals) > 1:
                agg[f"{k}_std"] = float(np.nanstd(vals))
        p2agg[label] = agg
    if "fno" in p2agg and "agfno" in p2agg and "agfno_frozen" in p2agg:
        p2agg["_controls"] = {
            "frozen_vs_fno_T": p2agg["agfno_frozen"]["rel_l2_T_mean"] / p2agg["fno"]["rel_l2_T_mean"],
            "mechanism_gain_T": p2agg["agfno"]["rel_l2_T_mean"] / p2agg["agfno_frozen"]["rel_l2_T_mean"],
            "mechanism_gain_ring_T": p2agg["agfno"]["rel_l2_ring_T_mean"] / p2agg["agfno_frozen"]["rel_l2_ring_T_mean"],
            "mechanism_gain_2T": p2agg["agfno"]["rel_l2_2T_mean"] / p2agg["agfno_frozen"]["rel_l2_2T_mean"],
            "fno_mechanism_gain_T": p2agg["fno"]["rel_l2_T_mean"] / p2agg["agfno"]["rel_l2_T_mean"],
            "fno_mechanism_gain_ring_T": p2agg["fno"]["rel_l2_ring_T_mean"] / p2agg["agfno"]["rel_l2_ring_T_mean"],
            "fno_mechanism_gain_2T": p2agg["fno"]["rel_l2_2T_mean"] / p2agg["agfno"]["rel_l2_2T_mean"],
        }
    summary["pde2"] = p2agg
    _figure_pde2(p2agg, os.path.join(exp_root, "pde2_bars.png"))

    # ================= B. Extended forgetting sweep (depths 4/6/8/16) ====== #
    d_epochs = 3 if quick else C.EXPERIMENT.depth_epochs
    cfg_depth = replace(C.TRAIN, num_epochs=d_epochs)
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
    depths = (4,) if quick else DEPTHS
    depth_res = {}
    for depth in depths:
        for label, name in (("fno", "fno"), ("agfno", "agfno")):
            tag = f"d{depth}_{label}"
            rd = os.path.join(exp_root, tag)
            t0 = time.time()
            model, _ = train_one(name, ddata, C.MODEL, cfg_depth, device, rd,
                                 quick, seed=C.SEED, n_blocks=depth)
            m = _eval_darcy(model, ddata, device)
            depth_res[tag] = {"depth": depth, "model": label, **m}
            print(f"[experiments2] {tag} ({(time.time()-t0)/60:.1f} min): " +
                  " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    summary["depth"] = depth_res
    _figure_depth(depth_res, os.path.join(exp_root, "forgetting_curve_ext.png"))

    # ================= C. Geo-FNO-style baseline on obstacle Darcy ========= #
    tag = "geofno_darcy"
    rd = os.path.join(exp_root, tag)
    t0 = time.time()
    model, _ = train_one("geofno", ddata, C.MODEL, cfg_depth, device, rd,
                         quick, seed=C.SEED)
    m = _eval_darcy(model, ddata, device)
    summary["geofno"] = m
    print(f"[experiments2] {tag} ({(time.time()-t0)/60:.1f} min): " +
          " ".join(f"{k}={v:.4f}" for k, v in m.items()))

    _save_json(os.path.join(exp_root, "summary.json"),
               {"summary": summary,
                "advec_config": {k: str(v) for k, v in C.ADVEC.__dict__.items()
                                 if not k.startswith("_")}})
    print("[experiments2] summary:")
    print(json.dumps(summary, indent=2))
    return {"run_dir": exp_root, "summary": summary}


def _eval_darcy(model, data: dict, device: str) -> dict:
    """Darcy test metrics (same protocol as experiments.py)."""
    te = data["test"]
    m = U.eval_metrics(model, te["x"], te["g"], te["u"], te["interior"], te["ring"])
    return m


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _figure_pde2(agg: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l for l in ("fno", "agfno_frozen", "agfno") if l in agg]
    keys = [("rel_l2_T_mean", "rel_l2_T_std", "one step (T)"),
            ("rel_l2_2T_mean", "rel_l2_2T_std", "zero-shot rollout (2T)")]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (mk, sk, leg) in enumerate(keys):
        vals = [agg[l].get(mk, np.nan) for l in labels]
        errs = [agg[l].get(sk, 0.0) for l in labels]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=3, label=leg)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("Advection-diffusion past obstacles: T vs zero-shot 2T")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _figure_depth(depth_res: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in (
        (axes[0], "rel_l2", "global rel $L^2$"),
        (axes[1], "rel_l2_ring", "near-boundary rel $L^2$"),
    ):
        for label, color in (("fno", "tab:red"), ("agfno", "tab:blue")):
            pts = sorted(
                ((v["depth"], v[key]) for v in depth_res.values() if v["model"] == label),
            )
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=color, label=label.upper() if label == "fno" else "AGF-NO")
        ax.set_xlabel("depth (blocks)")
        ax.set_ylabel(title)
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle("Geometric forgetting vs depth (obstacle Darcy)")
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
