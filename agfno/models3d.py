"""3-D generalization of the AGF-NO stack.

Mirrors ``models.py`` exactly (same conventions, same gate protocol, same
anti-forgetting design) with three structural changes:

* ``SpectralConv3d``    -- rfftn over the last three axes; four learnable
  complex weight tensors covering the (+d,+h), (-d,+h), (+d,-h) and (-d,-h)
  frequency quadrants of the two full axes (the last axis keeps only the
  rfft half-spectrum).
* ``coord_features``    -- Fourier features of 3-D coordinates (6 channels:
  sin/cos of 2*pi*x, y, z).
* physics               -- Darcy flow through 3-D obstacles (spheres and
  tori). Tori matter scientifically: a box with a solid-torus obstacle is
  not simply connected in an even stronger sense than the 2-D case, so
  deformation-based baselines (Geo-FNO family) remain structurally
  handicapped -- there is no diffeomorphism of the box that flattens an
  interior wall.

Everything else -- zero-initialized gates (AGF-NO starts as an exact FNO),
gate clamping, the frozen/spec_only control modes, the every-block SDF
re-injection, resolution-agnostic mode crop/pad for zero-shot
super-resolution -- is carried over verbatim so the 3-D experiment tests
the same mechanism, not a new one.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Standard FNO building block (3-D)
# --------------------------------------------------------------------------- #
class SpectralConv3d(nn.Module):
    """3-D spectral convolution: the core layer of a 3-D Fourier Neural Operator.

    Given v(x) with C_in channels on a D x H x W grid:

        1. V(k) = rFFTN(v)                    # over the last three axes
        2. keep only |k_i| <= modes_i (low frequencies; rest truncated)
        3. W_hat(k) is a learnable complex matrix acting channel-wise
        4. out(x) = iRFFTN( W_hat V )

    rfftn returns the full spectrum on the two leading transformed axes and
    the half-spectrum on the last, so the retained block is split into four
    quadrant weights (positive/negative frequency on each full axis). When
    modes_d == D the positive/negative slices touch row 0 (k = 0) twice; the
    second write wins -- identical to the canonical 2-D FNO behaviour.
    """

    def __init__(
        self, in_ch: int, out_ch: int, modes_d: int, modes_h: int, modes_w: int
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes_d = modes_d
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1.0 / (in_ch * out_ch)
        shape = (in_ch, out_ch, modes_d, modes_h, modes_w)
        self.weights1 = nn.Parameter(  # (+d, +h)
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(  # (-d, +h)
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(  # (+d, -h)
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(  # (-d, -h)
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: [B, C, D, H, W] (real)
        B, C, D, H, W = v.shape
        md = min(self.modes_d, D)
        mh = min(self.modes_h, H)
        mw = min(self.modes_w, W // 2 + 1)

        x_ft = torch.fft.rfftn(v, dim=(-3, -2, -1))  # [B, C, D, H, W//2+1]

        out_ft = torch.zeros(
            B, self.out_ch, D, H, W // 2 + 1, dtype=torch.cfloat, device=v.device
        )

        sd_p, sd_n = slice(0, md), slice(-md, None)
        sh_p, sh_n = slice(0, mh), slice(-mh, None)
        sw = slice(0, mw)

        for w, sl_d, sl_h in (
            (self.weights1, sd_p, sh_p),
            (self.weights2, sd_n, sh_p),
            (self.weights3, sd_p, sh_n),
            (self.weights4, sd_n, sh_n),
        ):
            out_ft[..., sl_d, sl_h, sw] = torch.einsum(
                "bixyz,ioxyz->boxyz",
                x_ft[..., sl_d, sl_h, sw].cfloat(),
                w[..., :md, :mh, :mw],
            )

        return torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1))


# --------------------------------------------------------------------------- #
# Geometry-aware spectral block (3-D) -- the contribution, verbatim in 3-D
# --------------------------------------------------------------------------- #
class AGFSpectralBlock3d(nn.Module):
    """3-D geometry-aware spectral block; identical math to the 2-D one:

        y = act( (W v) * (1 + g_s * tanh(S(W_s sdf))) + W_p v
                 + g_m * MLP_geo([h, sdf, coords]) )

    with both gates zero-initialized (exact FNO block at init) and clamped
    to ``gate_max``. The raw SDF channels re-enter the pointwise branch at
    every block, so boundary information never has to survive the Markovian
    chain of global mixers (the geometric-forgetting failure mode).
    """

    def __init__(
        self,
        width: int,
        modes_d: int,
        modes_h: int,
        modes_w: int,
        sdf_ch: int,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        gate_max: float = 5.0,
        use_coord_features: bool = True,
        n_coord_features: int = 6,
    ):
        super().__init__()
        self.width = width
        self.gate_max = gate_max

        # --- exact FNO sub-block ------------------------------------------
        self.conv = SpectralConv3d(width, width, modes_d, modes_h, modes_w)
        self.lin = nn.Conv3d(width, width, 1)
        hidden = mlp_ratio * width
        self.mlp = nn.Sequential(
            nn.Conv3d(width, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden, width, 1),
        )

        # --- geometry-aware additions (gated, zero-initialized) -----------
        self.sdf_conv = SpectralConv3d(sdf_ch, width, modes_d, modes_h, modes_w)
        self.sdf_lin = nn.Conv3d(sdf_ch, width, 1)
        self.gate_spec = nn.Parameter(torch.zeros(1))
        extra = sdf_ch + (n_coord_features if use_coord_features else 0)
        self.mlp_geo = nn.Sequential(
            nn.Conv3d(width + extra, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden, width, 1),
        )
        self.gate_mlp = nn.Parameter(torch.zeros(1))

        self.extra = extra
        self.use_coord_features = use_coord_features
        self.n_coord_features = n_coord_features

    @staticmethod
    def coord_features(B: int, D: int, H: int, W: int, device, n: int) -> torch.Tensor:
        """Fixed Fourier-feature encoding of the unit-cube coordinates."""
        zs = torch.linspace(0.0, 1.0, D, device=device)
        ys = torch.linspace(0.0, 1.0, H, device=device)
        xs = torch.linspace(0.0, 1.0, W, device=device)
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        feats = [
            torch.sin(2 * math.pi * gx),
            torch.cos(2 * math.pi * gx),
            torch.sin(2 * math.pi * gy),
            torch.cos(2 * math.pi * gy),
            torch.sin(2 * math.pi * gz),
            torch.cos(2 * math.pi * gz),
        ][:n]
        return torch.stack(feats, dim=0)[None].expand(B, -1, -1, -1, -1)

    def forward(
        self, v: torch.Tensor, sdf: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 1. Primary spectral branch, modulated by a learned geometric field.
        g_spec = self.sdf_conv(sdf) + self.sdf_lin(sdf)
        gs = self.gate_spec.clamp(-self.gate_max, self.gate_max)
        v_geo = self.conv(v) * (1.0 + gs * torch.tanh(g_spec))

        # 2. Pointwise linear branch -- exactly as in FNOBlock3d.
        h = F.gelu(v_geo + self.lin(v))  # [B, width, D, H, W]

        # 3. MLP branch + gated geometry re-injection (anti-forgetting skip).
        if self.use_coord_features:
            if coords is None:
                B, _, D, H, W = h.shape
                coords = self.coord_features(
                    B, D, H, W, h.device, self.n_coord_features
                )
            h_in = torch.cat([h, sdf, coords], dim=1)
        else:
            h_in = torch.cat([h, sdf], dim=1)
        gm = self.gate_mlp.clamp(-self.gate_max, self.gate_max)
        h = h + self.mlp(h) + gm * self.mlp_geo(h_in)
        return F.gelu(h)


class FNOBlock3d(nn.Module):
    """Standard 3-D FNO residual block (baseline)."""

    def __init__(
        self,
        width: int,
        modes_d: int,
        modes_h: int,
        modes_w: int,
        sdf_ch: int,  # unused; interface parity
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        hidden = mlp_ratio * width
        self.conv = SpectralConv3d(width, width, modes_d, modes_h, modes_w)
        self.lin = nn.Conv3d(width, width, 1)
        self.mlp = nn.Sequential(
            nn.Conv3d(width, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden, width, 1),
        )

    def forward(
        self, v: torch.Tensor, sdf: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = F.gelu(self.conv(v) + self.lin(v))
        h = h + self.mlp(h)
        return F.gelu(h)


# --------------------------------------------------------------------------- #
# Full 3-D operator backbones
# --------------------------------------------------------------------------- #
def _make_block3d(name: str, **kw):
    return AGFSpectralBlock3d(**kw) if name == "agfno" else FNOBlock3d(**kw)


class Operator3D(nn.Module):
    """Shared lifting / block-stack / projection skeleton in 3-D.

    Input channels: [log K, sdf, ring]; geometry stream: [sdf, ring,
    interior]; output: u. Gate modes ("full" / "spec_only" / "frozen") are
    the same controlled-ablation mechanism as the 2-D ``Operator2D``.
    """

    def __init__(
        self,
        kind: str,  # "fno" | "agfno"
        in_ch: int,
        out_ch: int,
        width: int,
        n_blocks: int,
        modes_d: int,
        modes_h: int,
        modes_w: int,
        sdf_ch: int,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        gate_max: float = 5.0,
        use_coord_features: bool = True,
        n_coord_features: int = 6,
        gate_mode: str = "full",
    ):
        super().__init__()
        self.kind = kind
        self.width = width
        self.gate_mode = gate_mode
        self.lift = nn.Conv3d(in_ch, width, 1)
        self.blocks = nn.ModuleList(
            [
                _make_block3d(
                    kind,
                    width=width,
                    modes_d=modes_d,
                    modes_h=modes_h,
                    modes_w=modes_w,
                    sdf_ch=sdf_ch,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    gate_max=gate_max,
                    use_coord_features=use_coord_features,
                    n_coord_features=n_coord_features,
                )
                for _ in range(n_blocks)
            ]
        )
        hidden = 2 * width
        self.proj = nn.Sequential(
            nn.Conv3d(width, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, out_ch, 1),
        )

        if gate_mode == "spec_only":
            for blk in self.blocks:
                blk.gate_mlp.data.zero_()
                blk.gate_mlp.requires_grad_(False)
        elif gate_mode == "frozen":
            for blk in self.blocks:
                blk.gate_spec.data.zero_()
                blk.gate_spec.requires_grad_(False)
                blk.gate_mlp.data.zero_()
                blk.gate_mlp.requires_grad_(False)

    def forward(self, a: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        """a: [B, in_ch, D, H, W]; sdf: [B, sdf_ch, D, H, W]."""
        v = self.lift(a)
        for blk in self.blocks:
            v = blk(v, sdf)
        return self.proj(v)


def fno3d(cfg) -> Operator3D:
    return Operator3D(
        kind="fno",
        in_ch=cfg.in_ch,
        out_ch=cfg.out_ch,
        width=cfg.width,
        n_blocks=cfg.n_blocks,
        modes_d=cfg.modes_d,
        modes_h=cfg.modes_h,
        modes_w=cfg.modes_w,
        sdf_ch=cfg.sdf_ch,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    )


def agfno3d(cfg, gate_mode: str = "full") -> Operator3D:
    return Operator3D(
        kind="agfno",
        in_ch=cfg.in_ch,
        out_ch=cfg.out_ch,
        width=cfg.width,
        n_blocks=cfg.n_blocks,
        modes_d=cfg.modes_d,
        modes_h=cfg.modes_h,
        modes_w=cfg.modes_w,
        sdf_ch=cfg.sdf_ch,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        gate_max=cfg.gate_max,
        use_coord_features=cfg.use_coord_features,
        n_coord_features=cfg.n_coord_features,
        gate_mode=gate_mode,
    )


def build_model3d(name: str, cfg, gate_mode: str = "full") -> Operator3D:
    name = name.lower()
    if name in ("fno", "baseline", "vanilla"):
        return fno3d(cfg)
    if name in ("agfno", "ours", "agf-no"):
        return agfno3d(cfg, gate_mode=gate_mode)
    raise ValueError(f"unknown 3-D model '{name}' (expected 'fno' or 'agfno')")
