"""A-priori diagnostic: will the geometry mechanism help on a given dataset?

The extended suite produced an honest asymmetry: the mechanism roughly halves
near-boundary error on obstacle Darcy (PDE1) but is near-neutral on
advection-diffusion (PDE2). Post-hoc, the explanation is "wall-anchored
difficulty". This module turns that explanation into a *prediction made
before training*.

v1 (naive) failed and the failure is informative: the raw truncated-band
fraction rho of the TARGET is larger for PDE2 than PDE1 -- but PDE2's high
band is inherited from its INPUT (the initial bumps are already zeroed
inside the obstacles), while PDE1's input is a smooth random field and the
*SOLVE* must create the wall-anchored high band. The band-limitation
theorem is about what the operator must CREATE, not what it must preserve.

The corrected statistic is therefore the change in truncated-band content
from input to target, ring-restricted:

    delta_rho = rho_ring(target) - rho_ring(input)

delta_rho > 0  =>  the task forces the network to create truncated-band
boundary content near walls, which the FNO's global channel structurally
cannot carry and the AGF modulation restores  =>  mechanism should pay.
delta_rho <= 0  =>  the high band is provided (or smoothed) by the dynamics
=>  nothing for the mechanism to restore.

Measured here: PDE1 delta_rho_ring ~ +0.024 (mechanism gain 0.445),
PDE2 delta_rho_ring ~ -0.024 (mechanism gain 0.965): equal magnitude,
opposite sign, matching the gains. rho is a ratio of spectral powers, hence
scale-invariant; no normalization is used.

Cost: a few FFTs per field -- computable on a candidate dataset in seconds,
before any training.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import config as C
from . import dataset as D


# --------------------------------------------------------------------------- #
# Core statistic
# --------------------------------------------------------------------------- #
def _band_power(x: torch.Tensor, kmax: float) -> dict:
    """Radial power split of [B, 1, H, W] fields above/below the mode cut.

    x is mean-removed per sample by the caller (DC excluded from both
    numerator and denominator; the DC bin carries no band information).
    """
    assert x.ndim == 4
    B, _, H, W = x.shape
    P = torch.fft.fft2(x).abs() ** 2
    ky = torch.fft.fftfreq(H, device=x.device)[:, None] * H
    kx = torch.fft.fftfreq(W, device=x.device)[None, :] * W
    k = torch.sqrt(ky**2 + kx**2)
    hi = (k > kmax).float()
    tot = P.sum(dim=(-2, -1)).clamp_min(1e-12)
    hi_e = (P * hi).sum(dim=(-2, -1))
    return {
        "hi_frac": hi_e / tot,        # [B, 1]
        "hi_power": hi_e[:, 0],       # [B]
        "tot_power": tot[:, 0],       # [B]
    }


@torch.no_grad()
def dataset_rho(u: torch.Tensor, ring: torch.Tensor, modes_h: int, modes_w: int,
                max_n: int = 256, batch: int = 64) -> dict:
    """rho statistics for a stack of target fields [N, 1, H, W]."""
    if u.ndim == 3:
        u = u[:, None]
    if ring.ndim == 3:
        ring = ring[:, None]
    kmax = float(max(modes_h, modes_w))
    N = min(u.shape[0], max_n)
    hi_fracs, ring_hi, ring_tot = [], [], []
    for i0 in range(0, N, batch):
        ub = u[i0 : i0 + batch].float()
        rb = ring[i0 : i0 + batch].float()
        ub = ub - ub.mean(dim=(-2, -1), keepdim=True)  # kill DC
        bp = _band_power(ub, kmax)
        hi_fracs.append(bp["hi_frac"][:, 0])
        rbm = _band_power(ub * rb, kmax)  # ring-masked field
        ring_hi.append(rbm["hi_power"])
        ring_tot.append(rbm["tot_power"])
    hi = torch.cat(hi_fracs)
    rh = torch.cat(ring_hi)
    rt = torch.cat(ring_tot).clamp_min(1e-12)
    return {
        "rho_global_mean": float(hi.mean()),
        "rho_global_std": float(hi.std()),
        "rho_ring_mean": float((rh / rt).mean()),
        "rho_ring_std": float((rh / rt).std()),
        "n_samples": int(N),
        "kmax": kmax,
    }


# --------------------------------------------------------------------------- #
# Per-PDE data plumbing
# --------------------------------------------------------------------------- #
def rho_for_darcy(vol_root: str, n: int, res: int) -> dict:
    """rho on the exact deterministic obstacle-Darcy test split."""
    te = D.load_or_make_split("test", n, res, vol_root, "cpu")
    _, ten, _ = D.normalize_split(te, te)  # stats self-consistent; rho invariant
    u = torch.tensor(np.asarray(ten["u"]), dtype=torch.float32)
    a = torch.tensor(np.asarray(ten["a"]), dtype=torch.float32)
    r = torch.tensor(np.asarray(ten["ring"]), dtype=torch.float32)
    out = dataset_rho(u, r, C.MODEL.modes_h, C.MODEL.modes_w)
    inp = dataset_rho(a, r, C.MODEL.modes_h, C.MODEL.modes_w)
    out["dataset"] = "pde1_darcy_obstacles"
    out["rho_input_ring_mean"] = inp["rho_ring_mean"]
    out["delta_rho_ring_mean"] = out["rho_ring_mean"] - inp["rho_ring_mean"]
    return out


def rho_for_advec(vol_root: str, n: int, res: int) -> dict:
    """rho on a fresh deterministic draw from the advection-diffusion generator."""
    d = D.make_advec_split("test", n, res, "cpu")
    u = torch.tensor(np.asarray(d["u_T"]), dtype=torch.float32)
    a = torch.tensor(np.asarray(d["a"]), dtype=torch.float32)
    r = torch.tensor(np.asarray(d["ring"]), dtype=torch.float32)
    out = dataset_rho(u, r, C.MODEL.modes_h, C.MODEL.modes_w)
    inp = dataset_rho(a, r, C.MODEL.modes_h, C.MODEL.modes_w)
    out["dataset"] = "pde2_advection_diffusion"
    out["rho_input_ring_mean"] = inp["rho_ring_mean"]
    out["delta_rho_ring_mean"] = out["rho_ring_mean"] - inp["rho_ring_mean"]
    return out


def kernel_main(artifacts_root: str = "/kaggle/working", n: int = 96) -> dict:
    """GPU-kernel entry: compute both rho values and write the JSON."""
    vol_root = (
        C.INFRA.volume_mount
        if os.path.isdir(C.INFRA.volume_mount)
        else "/tmp/agfno_vol"
    )
    os.makedirs(vol_root, exist_ok=True)
    out = {
        "modes": {"h": C.MODEL.modes_h, "w": C.MODEL.modes_w},
        "res": C.EXPERIMENT.res,
        "pde1_darcy": rho_for_darcy(vol_root, n, C.EXPERIMENT.res),
        "pde2_advec": rho_for_advec(vol_root, n, C.ADVEC.res),
    }
    out["rho_ring_ratio_pde1_over_pde2"] = (
        out["pde1_darcy"]["rho_ring_mean"]
        / max(out["pde2_advec"]["rho_ring_mean"], 1e-12)
    )
    os.makedirs(artifacts_root, exist_ok=True)
    with open(os.path.join(artifacts_root, "diagnostic_rho.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
# Local merge with the measured gains
# --------------------------------------------------------------------------- #
def run(root: str = "kaggle", rho_path: str | None = None) -> dict:
    """Merge rho (from the diag kernel output) with measured mechanism gains."""
    rp = rho_path or os.path.join(root, "diag_out", "diagnostic_rho.json")
    with open(rp) as f:
        rho = json.load(f)
    out = dict(rho)

    gains = {}
    p1 = os.path.join(root, "exp_out", "runs", "experiments", "summary.json")
    p2 = os.path.join(root, "exp2_out", "runs", "experiments2", "summary.json")
    if os.path.exists(p1):
        s = json.load(open(p1))["summary"]
        c = s.get("_controls", {})
        gains["pde1_ring_gain_vs_frozen"] = c.get("mechanism_gain_ring")
        gains["pde1_global_gain_vs_frozen"] = c.get("mechanism_gain_rel_l2")
    if os.path.exists(p2):
        s = json.load(open(p2))["summary"]
        c = s.get("pde2", {}).get("_controls", {})
        gains["pde2_ring_gain_vs_frozen"] = c.get("mechanism_gain_ring_T")
        gains["pde2_global_gain_vs_frozen"] = c.get("mechanism_gain_T")
    out["measured_gains"] = gains

    ratio = out["rho_ring_ratio_pde1_over_pde2"]
    d1 = out["pde1_darcy"].get("delta_rho_ring_mean")
    d2 = out["pde2_advec"].get("delta_rho_ring_mean")
    g1 = gains.get("pde1_ring_gain_vs_frozen")
    g2 = gains.get("pde2_ring_gain_vs_frozen")
    if g1 is not None and g2 is not None and d1 is not None and d2 is not None:
        # gain < 1 means the mechanism beats its frozen self. The corrected
        # prediction: gains order with delta_rho (the CREATED truncated band).
        out["prediction_check"] = {
            "delta_rho_orders_with_gain": bool(d1 > d2 and g1 < g2),
            "delta_rho_pde1": d1,
            "delta_rho_pde2": d2,
            "PASS": bool(d1 > d2 and g1 < g2),
        }
        # keep the v1 check visible for honesty: it FAILS on this pair.
        out["naive_rho_check"] = {
            "naive_rho_orders_with_gain": bool(ratio > 1.5 and g1 < g2),
            "note": "v1 statistic (target rho alone) inverts on PDE1 vs PDE2; "
                    "retained as a documented negative result",
        }

    os.makedirs(os.path.join(root, "diagnostic_out"), exist_ok=True)
    with open(os.path.join(root, "diagnostic_out", "diagnostic.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="kaggle")
    ap.add_argument("--rho", default=None, help="path to diagnostic_rho.json")
    ap.add_argument("--kernel", action="store_true",
                    help="compute rho (use on the GPU kernel)")
    ap.add_argument("--n", type=int, default=96)
    a = ap.parse_args()
    if a.kernel:
        kernel_main("/kaggle/working", a.n)
    else:
        run(a.root, a.rho)
