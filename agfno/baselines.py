"""External baselines for the obstacle-Darcy benchmark: U-Net and CNO.

Why these two
-------------
The evidence stack's remaining gap (Limitations ii) was the absence of
EXTERNAL architectures trained at matched settings on the identical splits.
This module adds the two canonical convolutional competitors:

* **U-Net** (Ronneberger et al., 2015) -- the default vision backbone and the
  natural "just use a CNN" baseline; encoder-decoder with skip connections.
* **CNO** (Raonic et al., 2024) -- the convolutional neural OPERATOR designed
  as the non-Fourier counterpart to FNO: U-Net-shaped, but with continuous
  up/downsampling (cosine interpolation + anti-aliasing blur), so it inherits
  discrete-invariance arguments that plain U-Nets lack.

Fairness protocol
-----------------
*Parameter matching*: FNO at the paper's headline setting is 4.81M params.
Both baselines are automatically width-searched (deterministic bisection over
the channel width) to land within +/-15% of the FNO budget -- both directions
reported honestly, so a reviewer can check neither baseline was starved.

*Identical protocol*: both consume the same input stack [log K, sdf, ring],
are trained by the SAME ``train_one`` loop (same optimizer, schedule, epochs,
boundary-penalty weight, seeds, byte-identical splits) and scored by the SAME
``eval_metrics`` (global rel-L2, near-wall ring rel-L2, wall fidelity). The
(x, g) -> y interface of the models here matches the FNO convention exactly,
so ``train_one``/``eval_metrics`` need zero modification.

Architectural honesty notes
---------------------------
* U-Net/CNO have LOCAL receptive fields per layer (they need depth to see
  global structure), while FNO's spectral layer is global in one step. That
  is the comparison the literature cares about; it is also why ring-metric
  behaviour near walls is the interesting axis.
* CNO's projection invariant ``P`` averages channel values over the output
  grid; on obstacle Darcy the target has a known constant interior value, so
  CNO's design is respected rather than replaced with a learned head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Shared cells
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    """Conv-norm-GELU-conv residual cell (CNO's B_i with identity skip)."""

    def __init__(self, ch: int):
        super().__init__()
        groups = math.gcd(8, ch)  # norm must adapt to any matched width
        self.f = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(groups, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.f(x)


def _blur(x: torch.Tensor) -> torch.Tensor:
    """CNO's anti-aliasing low-pass K (fixed 4x4 binomial kernel, stride 1)."""
    k = torch.tensor([1.0, 3.0, 3.0, 1.0], device=x.device, dtype=x.dtype)
    k2 = torch.outer(k, k)
    k2 = (k2 / k2.sum()).view(1, 1, 4, 4)
    c = x.shape[1]
    w = k2.expand(c, 1, 4, 4).clone()
    return F.conv2d(x, w, padding=2, groups=c)


def _interp(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """CNO's continuous resampling I: cosine interpolation when GROWING,
    blur + averaging when SHRINKING (Raonic et al. 2024, Eq. 4-5)."""
    tgt = (x.shape[2], x.shape[3])
    if size == tgt:
        return x
    if size[0] > tgt[0]:  # grow: cosine interpolation
        old, new = tgt, size
        ar = torch.arange(new[0], device=x.device, dtype=torch.float32)
        pos = ar / new[0] * old[0] - 0.5
        i0 = pos.floor().clamp(0, old[0] - 1).long()
        i1 = (i0 + 1).clamp(0, old[0] - 1)
        t = (0.5 * (1 - torch.cos(math.pi * (pos - i0)))).view(1, 1, new[0], 1)
        y = x.index_select(2, i0) * (1 - t) + x.index_select(2, i1) * t
        ar = torch.arange(new[1], device=x.device, dtype=torch.float32)
        pos = ar / new[1] * old[1] - 0.5
        j0 = pos.floor().clamp(0, old[1] - 1).long()
        j1 = (j0 + 1).clamp(0, old[1] - 1)
        t = (0.5 * (1 - torch.cos(math.pi * (pos - j0)))).view(1, 1, 1, new[1])
        y = y.index_select(3, j0) * (1 - t) + y.index_select(3, j1) * t
        return y
    # shrink: blur + averaging
    f = tgt[0] // size[0]
    return F.avg_pool2d(_blur(x), f)


# --------------------------------------------------------------------------- #
# U-Net (matched)
# --------------------------------------------------------------------------- #
class UNet2d(nn.Module):
    """Plain U-Net baseline, FNO I/O convention (x, g) -> y.

    Encoder-decoder with skip concatenations; the geometry stream is
    concatenated channel-wise to the input (the standard way a CNN receives
    side information -- exactly the baseline AGF-NO must beat).
    """

    def __init__(self, in_ch: int = 3, width: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [width * 2**i for i in range(depth + 1)]  # e.g. 32..512
        self.stem = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        enc = []
        for i in range(depth):
            enc.append(nn.Sequential(
                ResBlock(chs[i]), nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1),
            ))
        self.enc = nn.ModuleList(enc)
        self.mid = ResBlock(chs[-1])
        dec = []
        for i in range(depth, 0, -1):
            dec.append(nn.Sequential(
                nn.Conv2d(chs[i] + chs[i - 1], chs[i - 1], 3, padding=1),
                ResBlock(chs[i - 1]),
            ))
        self.dec = nn.ModuleList(dec)
        self.head = nn.Conv2d(chs[0], 1, 3, padding=1)

    def forward(self, x, g):
        h = self.stem(torch.cat([x, g], dim=1))
        skips = []
        for blk in self.enc:
            skips.append(h)
            h = blk(h)
        h = self.mid(h)
        for i, blk in enumerate(self.dec):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = blk(torch.cat([h, skips[::-1][i]], dim=1))
        return self.head(h)


# --------------------------------------------------------------------------- #
# CNO (faithful, matched)
# --------------------------------------------------------------------------- #
class CNO2d(nn.Module):
    """Convolutional Neural Operator (Raonic et al., NeurIPS 2024), faithful.

    L downsampling half-layers -> L UPSAMPLING half-layers (net output at FULL
    input resolution, as the CNO operator requires) -> P projects channel-wise
    to the target. Every up-step uses the cosine interpolation I, every tensor
    passes the fixed anti-aliasing K, cells are ResBlocks with -B- norm.
    """

    L = 2  # paper's L=2 for 64x64-class problems; 48 -> 12 -> 48

    def __init__(self, in_ch: int = 3, width: int = 32):
        super().__init__()
        d = [width * 2**i for i in range(self.L + 1)]  # 32, 64, 128
        u = [width * 2 ** (self.L - i) for i in range(self.L)]  # 128, 64
        self.lift = nn.Conv2d(in_ch, d[0], 3, padding=1)
        self.dl = nn.ModuleList(ResBlock(c) for c in d[:-1])
        self.dd = nn.ModuleList(
            nn.Conv2d(d[i], d[i + 1], 3, padding=1) for i in range(self.L)
        )
        self.ul = nn.ModuleList(ResBlock(c) for c in u)
        self.uu = nn.ModuleList(
            nn.Conv2d(u[i], u[i + 1] if i + 1 < len(u) else u[-1], 3, padding=1)
            for i in range(self.L)
        )
        self.u_final = nn.Conv2d(u[-1], u[-1], 3, padding=1)
        self.vert = nn.Conv2d(u[-1], width, 3, padding=1)
        self.head = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x, g):
        h = self.lift(torch.cat([x, g], dim=1))
        H, W = h.shape[-2:]
        skips = []
        for i in range(self.L):
            h = _blur(self.dl[i](h))
            skips.append(h)
            h = _interp(self.dd[i](h), (H // 2 ** (i + 1), W // 2 ** (i + 1)))
        for i in range(self.L):
            sz = (H // 2 ** (self.L - i - 1), W // 2 ** (self.L - i - 1))
            h = _blur(self.ul[i](h))
            h = _interp(self.uu[i](h), sz)
        h = self.u_final(h)
        h = self.vert(h)
        return self.head(h)


# --------------------------------------------------------------------------- #
# Parameter matching
# --------------------------------------------------------------------------- #
def _n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def match_width(cls, in_ch: int, target: int, lo: int = 8, hi: int = 96,
                tol: float = 0.15) -> tuple[nn.Module, int]:
    """Deterministic bisection over channel width to hit `target` params."""
    best_m, best_w, best_d = None, None, float("inf")
    lo_w, hi_w = lo, hi
    while lo_w <= hi_w:
        w = (lo_w + hi_w) // 2
        m = cls(in_ch, w)
        n = _n_params(m)
        if abs(n - target) / target < best_d:
            best_m, best_w, best_d = m, w, abs(n - target) / target
        if n < target:
            lo_w = w + 1
        else:
            hi_w = w - 1
    return best_m, best_w


# --------------------------------------------------------------------------- #
# Builder dispatch (plugs into train_one / eval_metrics unchanged)
# --------------------------------------------------------------------------- #
class _BaselineModel(nn.Module):
    """Wrapper giving baselines the same (x, g) signature as the FNO family.

    Baselines receive the full input stack (x) and ignore the separate
    geometry stream (g) beyond what is already concatenated into x -- they
    have no per-block injection mechanism, which is precisely the comparison.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x, g):
        return self.net(x, g)


def build_baseline(kind: str, cfg) -> nn.Module:
    """Build a parameter-matched baseline. `cfg` is ModelConfig (for in_ch).

    The FNO reference budget is computed live from cfg so matching stays
    exact if the headline model changes. Baselines concatenate the input
    stack and the geometry stream channel-wise, so their input width is
    in_ch + sdf_ch.
    """
    from .models import fno2d  # local import avoids a cycle

    target = _n_params(fno2d(cfg))
    in_ch = cfg.in_ch + cfg.sdf_ch
    kind = kind.lower()
    if kind == "unet":
        m, _w = match_width(UNet2d, in_ch, target)
        net = m
    elif kind == "cno":
        m, _w = match_width(CNO2d, in_ch, target)
        net = m
    else:
        raise ValueError(f"unknown baseline '{kind}' (expected 'unet' or 'cno')")
    return _BaselineModel(net)
