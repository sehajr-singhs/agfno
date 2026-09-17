"""Training utilities: losses, metrics, evaluation, and figures."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def _spatial_dims(x: torch.Tensor) -> tuple:
    """Spatial dimensions of a batched field tensor [B, C, *spatial]."""
    return tuple(range(2, x.ndim))


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batch-mean relative L2 error: ||p - t||_2 / ||t||_2 per sample.

    Dimension-agnostic: works for [B, C, H, W] and [B, C, D, H, W] alike
    (identical values on 2-D inputs)."""
    dims = _spatial_dims(pred)
    num = torch.sqrt(((pred - target) ** 2).sum(dim=dims) + 1e-12)
    den = torch.sqrt((target**2).sum(dim=dims) + 1e-12)
    return (num / den).mean()


def boundary_loss(
    pred: torch.Tensor, target: torch.Tensor, interior: torch.Tensor, ring: torch.Tensor
) -> torch.Tensor:
    """Boundary-fidelity penalty, measured against the ground truth:

    * interior term: mean |pred - target| on obstacle-interior cells
      (targets the wall condition u = const there, in whatever normalized
      units the training data uses),
    * ring term: mean |grad (pred - target)| on the boundary ring, pushing
      the prediction to reproduce the sharp solution kink at the wall.

    Architecturally reinforced in AGF-NO by the geometry-aware spectral path.
    """
    err = pred - target
    l_int = (err.abs() * interior).sum() / interior.sum().clamp_min(1.0)
    # Ring gradient term: sum of |partial| along every spatial axis (for 2-D
    # this equals the previous |d/dx| + |d/dy| expression exactly).
    g = sum(
        torch.gradient(err, dim=d)[0].abs() for d in range(2, err.ndim)
    )
    l_ring = (g * ring).sum() / ring.sum().clamp_min(1.0)
    return l_int + 0.5 * l_ring


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_metrics(
    model,
    a_all: torch.Tensor,
    sdf_all: torch.Tensor,
    u_all: torch.Tensor,
    interior_all: torch.Tensor,
    ring_all: torch.Tensor,
    batch: int = 64,
    max_n: int | None = None,
) -> dict:
    """Relative L2 (global / near-boundary) + wall-violation metrics.

    All inputs are GPU-resident tensors of shape [N, C, H, W]; batching is
    done manually to avoid dataloader overhead. near-boundary error is
    computed on the boundary ring (1.5 cells), which is where geometry-aware
    operators should shine.
    """
    model.eval()
    dims = _spatial_dims(a_all)
    tot = {"rel_l2": 0.0, "rel_l2_ring": 0.0, "wall_viol": 0.0, "n": 0}
    N = a_all.shape[0] if max_n is None else min(max_n, a_all.shape[0])
    for i in range(0, N, batch):
        a = a_all[i : i + batch]
        sdf = sdf_all[i : i + batch]
        u = u_all[i : i + batch]
        itl = interior_all[i : i + batch]
        rg = ring_all[i : i + batch]
        pred = model(a, sdf)
        rel = ((pred - u) ** 2).sum(dim=dims) / (u**2).sum(dim=dims).clamp_min(1e-12)
        rel = torch.sqrt(rel + 1e-12)
        # ring-weighted relative error (per-sample)
        num = (((pred - u) ** 2) * rg).sum(dim=dims)
        den = ((u**2) * rg).sum(dim=dims).clamp_min(1e-12)
        rel_ring = torch.sqrt(num / den + 1e-12)
        # wall fidelity: mean |pred - target| inside obstacles (the interior
        # ground truth is a constant in normalized units, NOT zero)
        wv = ((pred - u).abs() * itl).sum(dim=dims) / itl.sum(dim=dims).clamp_min(1.0)
        tot["rel_l2"] += rel.sum().item()
        tot["rel_l2_ring"] += rel_ring.sum().item()
        tot["wall_viol"] += wv.sum().item()
        tot["n"] += a.shape[0]
    n = max(tot["n"], 1)
    return {k: tot[k] / n for k in ("rel_l2", "rel_l2_ring", "wall_viol")}


# --------------------------------------------------------------------------- #
# Super-resolution: evaluate a model trained at res_lo on res_hi inputs
# --------------------------------------------------------------------------- #
@torch.no_grad()
def super_res_eval(model, a_hi, sdf_hi, u_hi, device: str, batch: int = 32) -> dict:
    """Zero-shot super-resolution: same physical samples on a 2x finer grid."""
    model.eval()
    rels = []
    for i in range(0, a_hi.shape[0], batch):
        a = a_hi[i : i + batch].to(device)
        sdf = sdf_hi[i : i + batch].to(device)
        u = u_hi[i : i + batch].to(device)
        pred = model(a, sdf)
        rel = ((pred - u) ** 2).sum(dim=_spatial_dims(u)) / (u**2).sum(
            dim=_spatial_dims(u)
        ).clamp_min(1e-12)
        rels.append(torch.sqrt(rel + 1e-12).cpu())
    rel = torch.cat(rels).mean().item()
    return {"rel_l2_sr": rel}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_comparison_figure(
    a: np.ndarray,
    sdf: np.ndarray,
    u_true: np.ndarray,
    u_fno: np.ndarray,
    u_ours: np.ndarray,
    path: str,
) -> None:
    """Ground truth vs FNO vs AGF-NO for one sample, with error maps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # accept [B, C, H, W] or [C, H, W] arrays; keep [H, W]
    u_true = np.asarray(u_true)[0, 0]
    u_fno = np.asarray(u_fno)[0, 0]
    u_ours = np.asarray(u_ours)[0, 0]

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 8.2))
    mats = [u_true, u_fno, u_ours]
    titles = ["Ground truth (PCG solver)", "FNO baseline", "AGF-NO (ours)"]
    vmin, vmax = float(np.min(u_true)), float(np.max(u_true))
    for j in range(n_cols):
        im = axes[0, j].imshow(mats[j], cmap="magma", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(titles[j], fontsize=11)
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
        plt.colorbar(im, ax=axes[0, j], fraction=0.046)
    errs = [np.abs(mats[j] - u_true) for j in range(n_cols)]
    emax = max(e.max() for e in errs) + 1e-12
    for j in range(n_cols):
        im = axes[1, j].imshow(errs[j], cmap="viridis", vmin=0, vmax=emax)
        rel = np.linalg.norm(errs[j]) / (np.linalg.norm(u_true) + 1e-12)
        axes[1, j].set_title(f"|error|  (rel L2 = {rel:.3f})", fontsize=11)
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
        plt.colorbar(im, ax=axes[1, j], fraction=0.046)
    # overlay obstacle boundary on all panels (sdf may be [C, H, W] or [B, C, H, W])
    sdf2d = np.asarray(sdf)
    while sdf2d.ndim > 2:
        sdf2d = sdf2d[0]
    boundary = np.abs(sdf2d) < 0.02
    for ax in axes.ravel():
        ax.contour(boundary, levels=[0.5], colors="cyan", linewidths=0.8)
    fig.suptitle("Darcy flow with irregular obstacles: prediction quality", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_error_history_figure(history: dict, path: str) -> None:
    """Training curves for both models on one axis set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, hist in history.items():
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], label=f"{name} train")
        axes[1].plot(ep, [h["val_rel_l2"] for h in hist], label=f"{name} val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("relative L2 loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].set_title("Training loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("relative L2 error")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title("Validation error")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_sr_figure(
    results: dict, path: str
) -> None:
    """Bar chart: coarse vs fine resolution error for both models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    coarse = [results[k]["rel_l2"] for k in labels]
    fine = [results[k].get("rel_l2_sr", float("nan")) for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - 0.18, coarse, 0.36, label=f"train resolution")
    ax.bar(x + 0.18, fine, 0.36, label="2x super-resolution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("relative L2 error")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Zero-shot super-resolution (48 -> 96 grid)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
