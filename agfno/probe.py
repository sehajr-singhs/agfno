"""Geometric-forgetting probe: is the domain geometry still *in* the network?

Motivation
----------
The "geometric forgetting" hypothesis states that deep neural operators lose
access to the domain geometry as depth grows, because each layer is a
Markovian global mixer: information about the boundary must survive a chain
of mixing operations, and global mixing averages it away.

Prior work cites this; almost nobody *measures* it. This module does.

Method
------
For a trained operator with L blocks we tap the latent state h_l in
[B, C, H, W] after every block. At each depth l we then ask a deliberately
*linear* question:

    can the signed distance to the obstacle boundary, S(x, y), be recovered
    from h_l(x, y) by a single linear map (ridge regression, closed form)?

The probe is fitted on one held-out-from-training batch of pixels and scored
by R^2 on a disjoint batch, with per-channel feature standardization. Because
the read-out is linear and identical for every architecture and depth, R^2 is
a clean proxy for "how much boundary information is linearly accessible here".

Prediction of the forgetting hypothesis: R^2 decays with depth for the plain
FNO. Prediction of our anti-forgetting design: AGF-NO's R^2 stays high
because the raw SDF is re-injected into the pointwise branch at every block
(and the spectral path is modulated by an SDF-derived field), so the geometry
never has to survive the mixing chain.

This is the direct test of the mechanism; the depth-vs-error sweep in
``experiments.py`` is the behavioural consequence.

Usage (needs the depth-sweep checkpoints produced by the experiment suite):

    python -m agfno.probe --root <artifacts>/runs/experiments --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import numpy as np
import torch

from . import config as C
from . import dataset as D
from .models import build_model


# --------------------------------------------------------------------------- #
# Latent extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_latents(model, x: torch.Tensor, g: torch.Tensor, batch: int = 32):
    """Return the per-block latent states plus the SDF target.

    Returns (latents, sdf) where latents is a list of [N, C_l, H, W] tensors
    (one per block, in order) and sdf is [N, 1, H, W].
    """
    model.eval()
    outs: list[list[torch.Tensor]] = []
    N = x.shape[0]
    for i0 in range(0, N, batch):
        xb, gb = x[i0 : i0 + batch], g[i0 : i0 + batch]
        v = model.lift(xb)
        per_block = []
        for blk in model.blocks:
            v = blk(v, gb)
            per_block.append(v.detach().cpu())
        outs.append(per_block)
    n_blocks = len(outs[0])
    latents = [torch.cat([o[l] for o in outs], dim=0) for l in range(n_blocks)]
    return latents, g[:, :1].detach().cpu()  # channel 0 of g is the SDF


# --------------------------------------------------------------------------- #
# Linear probe (closed-form ridge on per-pixel features)
# --------------------------------------------------------------------------- #
def _standardize_fit(X: torch.Tensor):
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mu, sd


def linear_probe_geometry(
    h: torch.Tensor,
    sdf: torch.Tensor,
    n_fit: int = 200_000,
    n_eval: int = 100_000,
    ridge: float = 1e-3,
    seed: int = 0,
) -> dict:
    """R^2 of a single linear map features -> signed distance.

    h   : [N, C, *spatial] latent features (2-D or 3-D)
    sdf : [N, 1, *spatial] target signed distance
    """
    Cdim = h.shape[1]
    # channel-last permutation for any spatial rank: (N, *spatial, C)
    perm = (0,) + tuple(range(2, h.ndim)) + (1,)
    X = h.permute(perm).reshape(-1, Cdim).float()
    # sdf matches h's rank; its size-1 channel vanishes in reshape
    y = sdf.permute(perm).reshape(-1).float()

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g)
    n_fit = min(n_fit, X.shape[0] // 2)
    n_eval = min(n_eval, X.shape[0] - n_fit)
    idx_fit, idx_eval = perm[:n_fit], perm[n_fit : n_fit + n_eval]

    Xf, yf = X[idx_fit], y[idx_fit]
    mu, sd = _standardize_fit(Xf)
    Xf = (Xf - mu) / sd
    # Closed-form ridge: w = (X^T X + ridge I)^-1 X^T y  (with bias column).
    Xf1 = torch.cat([Xf, torch.ones(Xf.shape[0], 1)], dim=1)
    A = Xf1.T @ Xf1
    A = A + ridge * torch.eye(A.shape[0])
    b = Xf1.T @ yf
    w = torch.linalg.solve(A, b)

    Xe = (X[idx_eval] - mu) / sd
    Xe1 = torch.cat([Xe, torch.ones(Xe.shape[0], 1)], dim=1)
    ye = y[idx_eval]
    pred = Xe1 @ w
    ss_res = ((ye - pred) ** 2).sum()
    ss_tot = ((ye - ye.mean()) ** 2).sum().clamp_min(1e-12)
    return {
        "r2": float(1.0 - ss_res / ss_tot),
        "rmse": float(torch.sqrt(((ye - pred) ** 2).mean())),
        "sd_target": float(ye.std()),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _load_ckpt(model_name: str, depth: int, path: str, device: str):
    cfg = replace(C.MODEL, n_blocks=depth)
    model = build_model(model_name, cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def run_probe(root: str, out_dir: str, n_samples: int = 192) -> dict:
    """Probe every depth-sweep checkpoint under `root`, write json + figure."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    vol = C.INFRA.volume_mount if os.path.isdir(C.INFRA.volume_mount) else "/tmp/agfno_vol"
    te = D.load_or_make_split("test", max(n_samples, 64), C.DATA.res, vol, device)
    _, te, _ = D.normalize_split(te, te)
    # Model input is the 3-channel stack [a, sdf, ring]; the geometry stream is
    # [sdf, ring, interior]; the probe target is the raw SDF (g channel 0).
    x = torch.cat(
        [
            torch.tensor(te["a"], dtype=torch.float32),
            torch.tensor(te["sdf"], dtype=torch.float32),
            torch.tensor(te["ring"], dtype=torch.float32),
        ],
        dim=1,
    )
    g = torch.cat(
        [
            torch.tensor(te["sdf"], dtype=torch.float32),
            torch.tensor(te["ring"], dtype=torch.float32),
            torch.tensor(te["interior"], dtype=torch.float32),
        ],
        dim=1,
    )
    x, g = x[:n_samples].to(device), g[:n_samples].to(device)
    print(f"[probe] device={device} samples={x.shape[0]} res={C.DATA.res}")

    results = {}
    # Discover depths actually present under `root` (depth4_, d4_, ...); fall
    # back to the configured depth range if the layout is not recognized.
    present = set()
    for entry in os.listdir(root):
        for pre in ("depth", "d"):
            if entry.startswith(pre):
                tail = entry[len(pre):]
                num = tail.split("_")[0]
                if num.isdigit():
                    present.add(int(num))
    depths = sorted(present) if present else range(1, C.MODEL.n_blocks + 1)
    print(f"[probe] depths found under root: {depths}")
    for model_name in ("fno", "agfno"):
        curve = {}
        for depth in depths:
            # depth 4 lives in the main seed-0 run; deeper/lower in the sweep.
            cands = [
                os.path.join(root, f"depth{depth}_{model_name}", "final.pt"),
                os.path.join(root, f"{model_name}_s{C.SEED}", "final.pt"),
                # experiments2.py layout: d{depth}_{label}/{name}_best.pt
                os.path.join(root, f"d{depth}_{model_name}", f"{model_name}_best.pt"),
                os.path.join(root, f"depth{depth}_{model_name}", f"{model_name}_best.pt"),
            ]
            path = next((p for p in cands if os.path.exists(p)), None)
            if path is None:
                print(f"[probe] skip {model_name} depth {depth}: no checkpoint")
                continue
            model = _load_ckpt(model_name, depth, path, device)
            latents, sdf = extract_latents(model, x, g)
            per_depth = []
            for l, h in enumerate(latents):
                r = linear_probe_geometry(h, sdf)
                per_depth.append({"block": l + 1, **r})
                print(
                    f"[probe] {model_name} depth {depth} block {l+1}: "
                    f"R^2={r['r2']:.4f} rmse={r['rmse']:.4f}"
                )
            curve[depth] = per_depth
        results[model_name] = curve

    # ---------------- Aggregation: R^2 at the FINAL block vs depth ---------
    final_r2 = {"fno": {}, "agfno": {}}
    for m, curve in results.items():
        for d, per_depth in curve.items():
            if per_depth:
                final_r2[m][int(d)] = per_depth[-1]["r2"]
    decay = {}
    for m, pts in final_r2.items():
        ds = sorted(pts)
        if len(ds) >= 2:
            decay[m] = float(pts[ds[0]] - pts[ds[-1]])  # drop in R^2 from depth1 -> deepest

    summary = {
        "protocol": (
            "linear ridge probe (closed form) regressing the signed distance "
            "S(x,y) from the latent features of each block; fitted on one "
            "half of the pixels, scored (R^2) on a disjoint half; features "
            "standardized per channel."
        ),
        "final_block_r2_by_depth": final_r2,
        "r2_drop_depth1_to_deepest": decay,
        "per_block_r2": {m: {str(d): v for d, v in c.items()} for m, c in results.items()},
    }
    with open(os.path.join(out_dir, "forgetting_probe.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _figure(results, os.path.join(out_dir, "forgetting_probe.png"))
    print("[probe] summary:")
    print(json.dumps({"final_block_r2_by_depth": final_r2, "r2_drop": decay}, indent=2))
    return summary


def _figure(results: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {"fno": "#888", "agfno": "#d33"}
    # Panel 1: R^2 at the final block vs depth (the forgetting curve).
    for m, curve in results.items():
        ds = sorted(curve)
        r2 = [curve[d][-1]["r2"] for d in ds]
        axes[0].plot(ds, r2, "o-", color=colors[m], label="FNO baseline" if m == "fno" else "AGF-NO")
    axes[0].set_xlabel("number of Fourier blocks (depth)")
    axes[0].set_ylabel("linear decodability of geometry  ($R^2$)")
    axes[0].set_ylim(0, 1.02)
    axes[0].axhline(0.0, color="k", lw=0.6)
    axes[0].legend(fontsize=9)
    axes[0].set_title("Geometric forgetting: is the boundary still decodable?", fontsize=10)
    axes[0].grid(alpha=0.3)
    # Panel 2: R^2 after every block, for the deepest available depth.
    for m, curve in results.items():
        if not curve:
            continue
        d = max(curve)
        blocks = [p["block"] for p in curve[d]]
        r2 = [p["r2"] for p in curve[d]]
        axes[1].plot(blocks, r2, "o-", color=colors[m], label=f"{m} (depth {d})")
    axes[1].set_xlabel("block index")
    axes[1].set_ylabel("$R^2$ (geometry decodable)")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=9)
    axes[1].set_title("Within-network decay (deepest model)", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Linear geometry probe: boundary information vs network depth", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing depth*/checkpoints")
    ap.add_argument("--out", default="probe_out")
    ap.add_argument("--n", type=int, default=192)
    args = ap.parse_args()
    run_probe(args.root, args.out, n_samples=args.n)
