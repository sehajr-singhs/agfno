"""AGF-NO: Adaptive Geometry-Aware Fourier Neural Operator.

Reference implementations of

* ``SpectralConv2d``  -- the standard FNO spectral convolution
  (rfft2 -> truncate -> learnable complex 1x1 -> irfft2),
* ``AGFSpectralBlock`` -- our geometry-aware spectral block. The Signed
  Distance Field (SDF) of the obstacle geometry is pushed through its own
  spectral conv and enters the operator *multiplicatively*, producing an
  effective kernel K_eff(x, y) = W(u)(x, y) . S(u)(x, y) that adapts to the
  local boundary geometry while keeping the O(N log N) FFT scaling, and
* two full operator backbones, ``FNO2d`` (baseline) and ``AGFNO2d`` (ours),
  sharing identical lifting/projection so the comparison is controlled.

All modules are resolution-agnostic w.r.t. the spectral path: Fourier modes
are cropped or zero-padded to match any input grid, which is what enables
zero-shot super-resolution evaluation (train at 48, test at 96).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Standard FNO building block
# --------------------------------------------------------------------------- #
class SpectralConv2d(nn.Module):
    """2-D spectral convolution: the core layer of the Fourier Neural Operator.

    Given an input field v(x, y) with C_in channels:

        1. V(k) = rFFT(v)                     # move to frequency domain
        2. keep only |k| <= modes (low frequencies; high modes are truncated)
        3. W_hat(k) is a learnable complex matrix acting channel-wise
        4. out(x) = iRFFT( W_hat V )          # back to physical space

    The truncation both regularizes (smooth physics lives at low frequency)
    and makes the layer quasilinear: O(N log N) in the number of grid points.
    """

    def __init__(self, in_ch: int, out_ch: int, modes_h: int, modes_w: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1.0 / (in_ch * out_ch)
        # Learnable complex weights for the non-negative-frequency half of the
        # 2-D spectrum produced by rfft2 (shape: [C_in, modes_h, modes_w]).
        self.weights1 = nn.Parameter(
            scale
            * torch.randn(in_ch, out_ch, modes_h, modes_w, dtype=torch.cfloat)
        )
        # Second weight tensor for the negative-frequency branch (k_h < 0),
        # which rfft2 outputs separately; using both halves doubles the
        # expressivity at identical cost.
        self.weights2 = nn.Parameter(
            scale
            * torch.randn(in_ch, out_ch, modes_h, modes_w, dtype=torch.cfloat)
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: [B, C, H, W] (real)
        B, C, H, W = v.shape
        mh, mw = min(self.modes_h, H), min(self.modes_w, W // 2 + 1)

        x_ft = torch.fft.rfft2(v, dim=(-2, -1))  # [B, C, H, W//2+1] complex

        out_ft = torch.zeros(
            B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=v.device
        )

        # positive vertical frequencies 0..mh-1
        w1 = self.weights1[..., :mh, :mw]
        # negative vertical frequencies -mh+1..-1 (stored as last rows)
        w2 = self.weights2[..., :mh, :mw]

        # Positive vertical freqs 0..mh-1 and negative freqs -(mh)..-1.
        # (When mh == H the negative slice includes row 0, i.e. frequency 0
        # is written twice; the second write wins -- identical to the
        # canonical FNO reference implementation.)
        slice_pos = (slice(None), slice(None), slice(0, mh), slice(0, mw))
        slice_neg = (slice(None), slice(None), slice(-mh, None), slice(0, mw))

        out_ft[slice_pos] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[slice_pos].cfloat(), w1
        )
        out_ft[slice_neg] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[slice_neg].cfloat(), w2
        )

        v_out = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))
        return v_out


# --------------------------------------------------------------------------- #
# Geometry-aware spectral block (the contribution)
# --------------------------------------------------------------------------- #
class AGFSpectralBlock(nn.Module):
    """Geometry-aware spectral block.

    Standard FNO block:      y = act( W v + b(v) )
    AGF-NO block (ours):     y = act( (W v) * (1 + g * S(W_s sdf)) + b(v) )

    where

    * ``W``  is the primary spectral conv on the latent field,
    * ``W_s`` is a *geometry* spectral conv acting on the injected SDF/boundary
      channels -- it synthesizes a complex-valued geometric modulation field
      with the same O(N log N) machinery,
    * ``g``  is a learnable per-layer gate (init 0), so at initialization the
      block is *exactly* a standard FNO block and training decides how much
      geometry adaptation to use. The gate is clamped to ``gate_max``.

    Because the modulation is computed from smooth spectral features of the
    SDF, the effective kernel K_eff = W . S adapts to the obstacle boundary
    (e.g. amplifying response near the no-flow walls) without any local
    stencil or graph edges -- preserving the FFT's global receptive field.

    The "anti-forgetting" skip: the raw SDF channels are also concatenated
    into the pointwise MLP branch at *every* depth, so boundary information
    never has to survive a Markovian chain of global mixers (the geometric
    forgetting failure mode of deep operators).
    """

    def __init__(
        self,
        width: int,
        modes_h: int,
        modes_w: int,
        sdf_ch: int,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        gate_max: float = 5.0,
        use_coord_features: bool = True,
        n_coord_features: int = 4,
    ):
        super().__init__()
        self.width = width
        self.gate_max = gate_max

        # --- exact FNO sub-block (identical structure to FNOBlock) ---------
        self.conv = SpectralConv2d(width, width, modes_h, modes_w)
        self.lin = nn.Conv2d(width, width, 1)
        hidden = mlp_ratio * width
        self.mlp = nn.Sequential(
            nn.Conv2d(width, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, width, 1),
        )

        # --- geometry-aware additions (both gated, zero-initialized) -------
        # Spectral geometry path: sdf channels -> modulation field.
        self.sdf_conv = SpectralConv2d(sdf_ch, width, modes_h, modes_w)
        self.sdf_lin = nn.Conv2d(sdf_ch, width, 1)
        self.gate_spec = nn.Parameter(torch.zeros(1))
        # Pointwise geometry path: re-inject sdf (+coords) in the MLP branch.
        extra = sdf_ch + (n_coord_features if use_coord_features else 0)
        self.mlp_geo = nn.Sequential(
            nn.Conv2d(width + extra, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, width, 1),
        )
        self.gate_mlp = nn.Parameter(torch.zeros(1))

        self.extra = extra
        self.use_coord_features = use_coord_features
        self.n_coord_features = n_coord_features

    @staticmethod
    def coord_features(B: int, H: int, W: int, device, n: int) -> torch.Tensor:
        """Fixed Fourier-feature encoding of the unit-square coordinates."""
        ys = torch.linspace(0.0, 1.0, H, device=device)
        xs = torch.linspace(0.0, 1.0, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        feats = [
            torch.sin(2 * math.pi * gx),
            torch.cos(2 * math.pi * gx),
            torch.sin(2 * math.pi * gy),
            torch.cos(2 * math.pi * gy),
        ][:n]
        return torch.stack(feats, dim=0)[None].expand(B, -1, -1, -1)

    def forward(self, v: torch.Tensor, sdf: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Primary spectral branch, modulated by a learned geometric field.
        g_spec = self.sdf_conv(sdf) + self.sdf_lin(sdf)
        gs = self.gate_spec.clamp(-self.gate_max, self.gate_max)
        v_geo = self.conv(v) * (1.0 + gs * torch.tanh(g_spec))

        # 2. Pointwise linear branch -- exactly as in FNOBlock.
        h = F.gelu(v_geo + self.lin(v))  # [B, width, H, W]

        # 3. MLP branch + gated geometry re-injection (anti-forgetting skip).
        if self.use_coord_features:
            if coords is None:
                B, _, H, W = h.shape
                coords = self.coord_features(B, H, W, h.device, self.n_coord_features)
            h_in = torch.cat([h, sdf, coords], dim=1)
        else:
            h_in = torch.cat([h, sdf], dim=1)
        gm = self.gate_mlp.clamp(-self.gate_max, self.gate_max)
        h = h + self.mlp(h) + gm * self.mlp_geo(h_in)
        return F.gelu(h)


class FNOBlock(nn.Module):
    """Standard FNO residual block (baseline): spectral conv + pointwise MLP."""

    def __init__(
        self,
        width: int,
        modes_h: int,
        modes_w: int,
        sdf_ch: int,  # unused; kept for interface parity
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        hidden = mlp_ratio * width
        self.conv = SpectralConv2d(width, width, modes_h, modes_w)
        self.lin = nn.Conv2d(width, width, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(width, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, width, 1),
        )

    def forward(self, v: torch.Tensor, sdf: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        h = F.gelu(self.conv(v) + self.lin(v))
        h = h + self.mlp(h)
        return F.gelu(h)


# --------------------------------------------------------------------------- #
# Full operator backbones
# --------------------------------------------------------------------------- #
def _make_block(name: str, **kw):
    return AGFSpectralBlock(**kw) if name == "agfno" else FNOBlock(**kw)


class Operator2D(nn.Module):
    """Shared lifting / block-stack / projection skeleton.

    Input channels: [a(x, y), sdf(x, y), ring(x, y)] (3 channels).
    Output: solution field u(x, y) (1 channel).
    """

    def __init__(
        self,
        kind: str,  # "fno" | "agfno"
        in_ch: int,
        out_ch: int,
        width: int,
        n_blocks: int,
        modes_h: int,
        modes_w: int,
        sdf_ch: int,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        gate_max: float = 5.0,
        use_coord_features: bool = True,
        n_coord_features: int = 4,
        gate_mode: str = "full",  # "full" | "spec_only" | "frozen"
    ):
        super().__init__()
        self.kind = kind
        self.width = width
        self.gate_mode = gate_mode
        # Lifting: 1x1 conv from physics channels to latent width.
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList(
            [
                _make_block(
                    kind,
                    width=width,
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
            nn.Conv2d(width, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch, 1),
        )

        # ------------------------------------------------------------------
        # Gate-mode controls (for controlled ablation experiments)
        #
        #   "full"      : both geometry paths trainable (the AGF-NO mechanism)
        #   "spec_only" : the anti-forgetting MLP re-injection is frozen at 0;
        #                 only the spectral gating is active. Isolates the
        #                 contribution of the re-injection path.
        #   "frozen"    : BOTH gates frozen at 0 -> the network has all the
        #                 AGF-NO parameters but the geometry mechanism is
        #                 switched off, so it must behave exactly like the FNO
        #                 baseline. This is the capacity/mechanism control: it
        #                 shows that any AGF-NO gain comes from the mechanism,
        #                 not from the extra parameters.
        # ------------------------------------------------------------------
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
        """a: [B, in_ch, H, W] physics input; sdf: [B, sdf_ch, H, W]."""
        v = self.lift(a)
        for blk in self.blocks:
            v = blk(v, sdf)
        return self.proj(v)


def fno2d(cfg) -> Operator2d:
    return Operator2D(
        kind="fno",
        in_ch=cfg.in_ch,
        out_ch=cfg.out_ch,
        width=cfg.width,
        n_blocks=cfg.n_blocks,
        modes_h=cfg.modes_h,
        modes_w=cfg.modes_w,
        sdf_ch=cfg.sdf_ch,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    )


def agfno2d(cfg, gate_mode: str = "full") -> Operator2d:
    return Operator2D(
        kind="agfno",
        in_ch=cfg.in_ch,
        out_ch=cfg.out_ch,
        width=cfg.width,
        n_blocks=cfg.n_blocks,
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


def build_model(name: str, cfg, gate_mode: str = "full") -> Operator2d:
    name = name.lower()
    if name in ("fno", "baseline", "vanilla"):
        return fno2d(cfg)
    if name in ("agfno", "ours", "agf-no"):
        return agfno2d(cfg, gate_mode=gate_mode)
    if name in ("geofno", "deformfno", "geo-fno"):
        return geofno2d(cfg)
    if name in ("unet", "cno"):
        # External convolutional baselines (parameter-matched to fno2d(cfg));
        # built through baselines.py so every experiment inherits matching.
        from .baselines import build_baseline

        return build_baseline(name, cfg)
    raise ValueError(f"unknown model '{name}' (expected 'fno', 'agfno', 'geofno', 'unet', or 'cno')")


# --------------------------------------------------------------------------- #
# Geo-FNO-style baseline: learned deformation + standard FNO
#
# Follows Li et al., JMLR 2023 ("Fourier Neural Operator with Learned
# Deformations"): a small network maps (coords, geometry) -> a displacement
# field; the input is sampled on the deformed grid, pushed through a STANDARD
# FNO, and the output is mapped back.
#
# Structural note (an honest expectation, not a strawman): the deformation is
# a diffeomorphism of the box, but our benchmark's obstacle walls are interior
# Dirichlet boundaries -- a domain with interior holes is NOT simply
# connected, so no diffeomorphic deformation can flatten it. Deformations
# help with deformed *outer* boundaries (Geo-FNO's setting); interior walls
# are natively represented by SDF conditioning instead. Identity at init:
# the displacement head is zero-initialized, so DeformFNO starts as an exact
# FNO (unit-tested), matching the zero-gate protocol of AGF-NO.
# --------------------------------------------------------------------------- #
class DeformationField(nn.Module):
    """Small CNN: [coords(2), sdf(3)] -> smooth displacement field (2 channels)."""

    def __init__(self, sdf_ch: int, width: int = 32, n_layers: int = 4, scale: float = 0.15):
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = 2 + sdf_ch
        for _ in range(n_layers):
            layers.append(nn.Conv2d(ch_in, width, 3, padding=1))
            layers.append(nn.GELU())
            ch_in = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.scale = scale

    def forward(self, sdf: torch.Tensor) -> torch.Tensor:
        B, _, H, W = sdf.shape
        ys = torch.linspace(-1.0, 1.0, H, device=sdf.device)
        xs = torch.linspace(-1.0, 1.0, W, device=sdf.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gx, gy], dim=0)[None].expand(B, -1, -1, -1)
        d = self.head(self.body(torch.cat([coords, sdf], dim=1)))
        return self.scale * torch.tanh(d)  # smooth, bounded displacement


class DeformFNO(nn.Module):
    """Geo-FNO-style operator: learned deformation + standard FNO + undo."""

    def __init__(self, cfg):
        super().__init__()
        self.defor = DeformationField(cfg.sdf_ch)
        self.fno = Operator2D(
            kind="fno",
            in_ch=cfg.in_ch,
            out_ch=cfg.out_ch,
            width=cfg.width,
            n_blocks=cfg.n_blocks,
            modes_h=cfg.modes_h,
            modes_w=cfg.modes_w,
            sdf_ch=cfg.sdf_ch,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

    def _warp(self, f: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """Sample f at x + disp(x). f: [B, C, H, W]; disp: [B, 2, H, W]."""
        B, _, H, W = f.shape
        ys = torch.linspace(-1.0, 1.0, H, device=f.device)
        xs = torch.linspace(-1.0, 1.0, W, device=f.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([gx, gy], dim=-1)  # [H, W, 2] (x, y)
        grid = base[None] + disp.permute(0, 2, 3, 1)
        # align_corners=True: linspace(-1, 1) then lands EXACTLY on pixel
        # centers, so disp == 0 gives a perfect identity warp (unit-tested).
        # With the default False, -1/1 map to pixel *edges* and interior
        # samples blur between neighbours even at zero displacement.
        return torch.nn.functional.grid_sample(
            f, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def forward(self, a: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        disp = self.defor(sdf)
        a_w = self._warp(a, disp)
        u_w = self.fno(a_w, sdf)
        return self._warp(u_w, -disp)  # approximate inverse mapping


def geofno2d(cfg) -> DeformFNO:
    return DeformFNO(cfg)
