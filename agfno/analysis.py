"""Frequency-domain analysis of the truncation band-limitation (P1, P2).

Measurements here make the theory in ``THEORY.md`` empirical:

P1  --  perturb-vs-clean propagation: corrupt a clean latent/input inside an
        obstacle, run the trained operator, and measure where the damage
        shows up spatially (ring vs interior vs far field). Band-limitation
        predicts the FNO's reachable influence is confined and mis-shaped;
        AGF-NO's modulation escapes the sinc confinement.

P2  --  frequency-resolved error: radial-band (low/mid/high) x region
        (ring/interior/far) decomposition of the prediction error. The
        truncation argument predicts FNO's misfit piles up in the HIGH band
        near walls (the truncated kink), while AGF-NO reduces exactly that
        cell of the matrix.

Both analyses operate on trained checkpoints and need no retraining; they
run on CPU in minutes for a few hundred test samples.

Also provided: ``sdf_band_energies`` -- the premise of the band-restoration
argument (the SDF's spectrum is nonzero above the mode cut).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import torch

from . import config as C
from . import dataset as D
from .models import build_model


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _radial_bands(ky: torch.Tensor, kx: torch.Tensor) -> dict[str, torch.Tensor]:
    """Boolean masks for low/mid/high radial frequency bands.

    Bands are relative to the *retained* mode cut k_max of the architecture:
    'low'   |k| <= 0.5 k_max
    'mid'   0.5 k_max < |k| <= k_max
    'high'  |k| > k_max  (the truncated band -- excluded from the global
            channel of a plain FNO)
    """
    k = torch.sqrt(ky**2 + kx**2)
    kmax = max(ky.max().item(), 1e-8)
    return {
        "low": k <= 0.5 * kmax,
        "mid": (k > 0.5 * kmax) & (k <= kmax),
        "high": k > kmax,
    }


def _fft2_rfft(x: torch.Tensor) -> torch.Tensor:
    """rfft2 with the frequency grids matching torch.fft.fftfreq conventions."""
    return torch.fft.rfft2(x)


def _spec_power(field: torch.Tensor) -> torch.Tensor:
    """|FFT|^2 power spectrum of [..., H, W] (complex, full-grid convention)."""
    return torch.fft.fft2(field).abs() ** 2


def sdf_band_energies(sdf: torch.Tensor, modes_h: int, modes_w: int) -> dict:
    """Premise check: how much SDF energy lives above the mode cut?

    sdf: [N, 1, H, W]. Returns mean energy fraction per band.
    """
    H, W = sdf.shape[-2:]
    ky = torch.fft.fftfreq(H)[:, None] * H
    kx = torch.fft.fftfreq(W)[None, :] * W
    bands = _radial_bands(ky, kx)
    P = _spec_power(sdf)
    total = P.sum(dim=(-2, -1)).clamp_min(1e-12)
    out = {}
    for name, mask in bands.items():
        frac = (P * mask).sum(dim=(-2, -1)) / total
        out[name] = float(frac.mean())
    out["kmax_arch"] = modes_h
    out["kmax_grid"] = H // 2
    return out


# --------------------------------------------------------------------------- #
# P2: frequency-resolved error decomposition
# --------------------------------------------------------------------------- #
@torch.no_grad()
def frequency_resolved_error(
    model,
    x: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    ring: torch.Tensor,
    interior: torch.Tensor,
    modes_h: int,
    modes_w: int,
    max_n: int = 256,
    batch: int = 64,
) -> dict:
    """Band x region decomposition of the error spectrum.

    Returns {'ring': {low, mid, high}, 'interior': {...}, 'far': {...}} with
    each entry the mean error-energy fraction in that band, and the ratio
    high/low as a sharpness summary. The prediction: FNO has a HIGH ring
    fraction well above AGF-NO's.
    """
    model.eval()
    H, W = u.shape[-2:]
    ky = torch.fft.fftfreq(H)[:, None] * H
    kx = torch.fft.fftfreq(W)[None, :] * W
    bands = _radial_bands(ky, kx)

    regions = {
        "ring": (ring > 0.5).float(),
        "interior": (interior > 0.5).float(),
        "far": ((ring < 0.5) & (interior < 0.5)).float(),
    }

    N = min(x.shape[0], max_n)
    acc = {r: {b: 0.0 for b in bands} for r in regions}
    acc_ratio = {r: 0.0 for r in regions}
    n_done = 0
    for i0 in range(0, N, batch):
        xb = x[i0 : i0 + batch]
        gb = g[i0 : i0 + batch]
        ub = u[i0 : i0 + batch]
        pred = model(xb, gb)
        err = (pred - ub).squeeze(1)  # [B, H, W]
        P = _spec_power(err)
        for rname, rmask in regions.items():
            rb = rmask[i0 : i0 + batch].squeeze(1)
            # window the error power by the region mask (spatially local spectra)
            Pw = _spec_power(err * rb)
            tot = Pw.sum(dim=(-2, -1)).clamp_min(1e-12)
            for bname, bmask in bands.items():
                acc[rname][bname] += float(
                    ((Pw * bmask).sum(dim=(-2, -1)) / tot).sum()
                )
            hi = (Pw * bands["high"]).sum(dim=(-2, -1))
            lo = (Pw * bands["low"]).sum(dim=(-2, -1)).clamp_min(1e-12)
            acc_ratio[rname] += float((hi / lo).sum())
        n_done += xb.shape[0]

    out = {}
    for r in regions:
        fracs = {b: acc[r][b] / n_done for b in bands}
        fracs["high_over_low"] = acc_ratio[r] / n_done
        out[r] = fracs
    out["_meta"] = {"n": n_done, "res": H, "modes_h": modes_h, "modes_w": modes_w}
    return out


# --------------------------------------------------------------------------- #
# P1: perturb-vs-clean propagation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def perturbation_propagation(
    model,
    x: torch.Tensor,
    g: torch.Tensor,
    sdf: torch.Tensor,
    interior: torch.Tensor,
    n: int = 64,
    amp: float = 3.0,
    seed: int = 0,
) -> dict:
    """How far does a wall perturbation travel through the operator?

    Protocol: take clean inputs; add a smooth random bump (amplitude `amp`
    in normalized units) INSIDE the largest obstacle of each sample; run the
    operator on perturbed vs clean inputs; measure the induced output change
    in three regions: the perturbed obstacle's interior, its boundary ring
    (3 cells), and the far field. Also report the spatial decay profile --
    mean |delta| as a function of distance from the wall.

    Band-limitation predicts: FNO transmits a sinc-shaped, low-pass
    influence; AGF-NO transmits more, and with a sharper near-wall profile.
    """
    model.eval()
    torch.manual_seed(seed)
    n = min(n, x.shape[0])
    ring3 = _dilate((interior > 0.5).float(), 3) & ~(interior > 0.5)
    far = ~(interior > 0.5) & ~ring3

    deltas_int, deltas_ring, deltas_far = [], [], []
    profiles = []
    for i in range(n):
        itl = (interior[i, 0] > 0.5)
        if itl.sum() < 4:
            continue
        # largest connected component = the obstacle to perturb
        lab = _largest_component(itl)
        mask = lab.float()
        bnd = _dilate(mask[None, None], 2)[0, 0] > 0.5
        # smooth bump supported on the obstacle
        bump = _smooth_bump(mask, amp)
        x_p = x[i : i + 1].clone()
        x_p[0, 0] += bump  # channel 0 = log K (the field the PDE feels)
        y0 = model(x[i : i + 1], g[i : i + 1])
        y1 = model(x_p, g[i : i + 1])
        d = (y1 - y0)[0, 0]
        deltas_int.append(float(d[mask > 0.5].abs().mean()))
        deltas_ring.append(float(d[bnd].abs().mean()))
        if far[i, 0].any():
            deltas_far.append(float(d[far[i, 0]].abs().mean()))
        # radial decay profile from the obstacle boundary
        profiles.append(
            _radial_profile(d.abs(), bnd, max_dist=24)
        )
    prof = np.mean(np.stack(profiles), axis=0).tolist()
    return {
        "delta_interior_mean": float(np.mean(deltas_int)),
        "delta_ring_mean": float(np.mean(deltas_ring)),
        "delta_far_mean": float(np.mean(deltas_far)),
        "ring_over_far": float(np.mean(deltas_ring) / max(np.mean(deltas_far), 1e-9)),
        "decay_profile_wall_to_far": prof,
        "n": len(deltas_int),
        "amp": amp,
    }


# --------------------------------------------------------------------------- #
# Small array utilities (torch, batch-of-one friendly)
# --------------------------------------------------------------------------- #
def _dilate(mask: torch.Tensor, iters: int) -> torch.Tensor:
    m = mask
    for _ in range(iters):
        m = torch.nn.functional.max_pool2d(m, 3, stride=1, padding=1)
    return m > 0.5


def _largest_component(mask: torch.Tensor) -> torch.Tensor:
    """Largest 4-connected component of a bool mask [H, W] (simple BFS)."""
    import collections

    H, W = mask.shape
    seen = torch.zeros_like(mask)
    best = torch.zeros_like(mask)
    best_size = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not seen[i, j]:
                comp = torch.zeros_like(mask)
                q = collections.deque([(i, j)])
                seen[i, j] = True
                cells = []
                while q:
                    a, b = q.popleft()
                    comp[a, b] = True
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < H and 0 <= nb < W and mask[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True
                            q.append((na, nb))
                if len(cells) > best_size:
                    best_size = len(cells)
                    best = comp
    return best


def _smooth_bump(mask: torch.Tensor, amp: float) -> torch.Tensor:
    """Smooth nonnegative bump supported (softly) on the obstacle mask."""
    m = mask.float()[None, None]
    # blur the indicator, renormalize, restrict
    k = torch.tensor([1.0, 2.0, 1.0])
    kx = (k[:, None] @ k[None, :]).reshape(1, 1, 3, 3)
    kx = kx / kx.sum()
    z = torch.nn.functional.conv2d(m, kx, padding=1)
    z = z / z.max().clamp_min(1e-6)
    return (z[0, 0] * mask.float()) * amp


def _radial_profile(
    d: torch.Tensor, bnd: torch.Tensor, max_dist: int = 24
) -> np.ndarray:
    """Mean |d| in distance shells from the obstacle boundary."""
    # distance transform via repeated dilation (cheap, good enough for shells)
    H, W = d.shape
    cur = bnd[None, None].float()
    sums = torch.zeros(max_dist)
    cnts = torch.zeros(max_dist)
    assigned = torch.zeros(H, W, dtype=torch.bool)
    assigned |= bnd
    for dist in range(max_dist):
        nxt = torch.nn.functional.max_pool2d(cur, 3, stride=1, padding=1)
        shell = (nxt[0, 0] > 0.5) & ~assigned
        sums[dist] = d[shell].sum()
        cnts[dist] = shell.sum()
        assigned |= shell
        cur = nxt
    prof = (sums / cnts.clamp_min(1)).numpy()
    return prof


# --------------------------------------------------------------------------- #
# Driver: run the full analysis on pulled checkpoints
# --------------------------------------------------------------------------- #
def run_analysis(
    ckpt_root: str,
    out_dir: str,
    n_samples: int = 256,
    n_perturb: int = 64,
) -> dict:
    """Frequency-resolved error (P2) + propagation (P1) for fno vs agfno.

    `ckpt_root` must contain fno_s0/final.pt and agfno_s0/final.pt (the
    controlled-experiment suite naming), and the test data is regenerated
    deterministically as everywhere else in the package.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    vol = C.INFRA.volume_mount if os.path.isdir(C.INFRA.volume_mount) else "/tmp/agfno_vol"
    te = D.load_or_make_split("test", 1000, C.DATA.res, vol, device)
    _, te, _ = D.normalize_split(te, te)

    x = torch.cat(
        [torch.tensor(te[k], dtype=torch.float32) for k in ("a", "sdf", "ring")], dim=1
    )
    g = torch.cat(
        [torch.tensor(te[k], dtype=torch.float32) for k in ("sdf", "ring", "interior")], dim=1
    )
    u = torch.tensor(te["u"], dtype=torch.float32)
    ring = torch.tensor(te["ring"], dtype=torch.float32)
    interior = torch.tensor(te["interior"], dtype=torch.float32)
    sdf = torch.tensor(te["sdf"], dtype=torch.float32)

    x, g, u = x[:n_samples].to(device), g[:n_samples].to(device), u[:n_samples].to(device)
    ring, interior = ring[:n_samples].to(device), interior[:n_samples].to(device)
    sdf = sdf[:n_samples].to(device)

    # Premise of the band-restoration argument
    premise = sdf_band_energies(sdf, C.MODEL.modes_h, C.MODEL.modes_w)
    print(f"[analysis] SDF band energies (premise): {premise}")

    results: dict = {"sdf_band_energies": premise, "models": {}}
    for name in ("fno", "agfno"):
        path = os.path.join(ckpt_root, f"{name}_s0", "final.pt")
        if not os.path.exists(path):
            print(f"[analysis] missing {path}; skipping {name}")
            continue
        model = build_model(name, C.MODEL).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        p2 = frequency_resolved_error(
            model, x, g, u, ring, interior, C.MODEL.modes_h, C.MODEL.modes_w
        )
        p1 = perturbation_propagation(model, x, g, sdf, interior, n=n_perturb)
        results["models"][name] = {"freq_resolved_error": p2, "propagation": p1}
        print(f"[analysis] {name}: P2 ring={p2['ring']} ")
        print(f"[analysis] {name}: P1 ring/far={p1['ring_over_far']:.2f}")

    with open(os.path.join(out_dir, "frequency_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    _figure(results, os.path.join(out_dir, "frequency_analysis.png"))
    print(f"[analysis] wrote {out_dir}/frequency_analysis.json")
    return results


def _figure(results: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in ("fno", "agfno") if m in results.get("models", {})]
    if not models:
        return
    colors = {"fno": "#888", "agfno": "#d33"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # Panel 1: P2 -- error energy fraction per band, ring region
    bands = ["low", "mid", "high"]
    w = 0.35
    for i, m in enumerate(models):
        fr = [results["models"][m]["freq_resolved_error"]["ring"][b] for b in bands]
        axes[0].bar(np.arange(3) + (i - 0.5) * w, fr, w, color=colors[m],
                    label="FNO" if m == "fno" else "AGF-NO")
    axes[0].set_xticks(np.arange(3))
    axes[0].set_xticklabels(["low band", "mid band", "HIGH band (truncated)"])
    axes[0].set_ylabel("error energy fraction (ring)")
    axes[0].set_yscale("log")
    axes[0].set_title("P2: where does the ring misfit live?", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: P1 -- propagation contrast
    metrics = ["delta_interior_mean", "delta_ring_mean", "delta_far_mean"]
    labels = ["inside\nperturbed obstacle", "boundary\nring", "far field"]
    for i, m in enumerate(models):
        vals = [results["models"][m]["propagation"][k] for k in metrics]
        axes[1].bar(np.arange(3) + (i - 0.5) * w, vals, w, color=colors[m],
                    label="FNO" if m == "fno" else "AGF-NO")
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("mean |Δoutput| from wall perturbation")
    axes[1].set_yscale("log")
    axes[1].set_title("P1: how far does a wall perturbation travel?", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: P1 -- spatial decay profiles
    for m in models:
        prof = results["models"][m]["propagation"]["decay_profile_wall_to_far"]
        axes[2].plot(prof, color=colors[m], label="FNO" if m == "fno" else "AGF-NO")
    axes[2].set_xlabel("distance from wall (cells)")
    axes[2].set_ylabel("mean |Δoutput|")
    axes[2].set_yscale("log")
    axes[2].set_title("Influence decay from the wall", fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        "Truncation band-limitation: frequency-resolved evidence (P1, P2)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--out", default="analysis_out")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--n-perturb", type=int, default=64)
    a = ap.parse_args()
    run_analysis(a.ckpt_root, a.out, n_samples=a.n, n_perturb=a.n_perturb)
