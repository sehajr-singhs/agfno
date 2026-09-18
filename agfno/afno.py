"""AFNO-style architecture: the third member of the band-limited global family.

Adaptive Fourier Neural Operator (Guibas et al., ICLR 2022; as deployed in
FourCastNet, Pathak et al. 2022) replaces FNO's *hard* mode truncation with a
*soft* shrinkage of the spectrum and a pre-norm residual design. Like the FNO,
it is a periodic-domain global mixer with a band-limited spectral channel --
exactly the family whose geometric behaviour this paper measures -- and its
depth-dependent geometric forgetting has never been tested.

Two variants are provided, both on the shared ``Operator2D`` skeleton (same
lifting/projection as FNO/AGF-NO, so parameter counts and the probe interface
line up exactly):

* ``afno``  -- the faithful mixer at the FNO's mode budget: real-FFT,
  block-diagonal complex weights on the retained band, soft-threshold spectrum
  shrinkage (complex ``softshrink`` with threshold ``sparsity_th``), shared
  complex channel mixing (``chan_relaxed``), pre-norm residual stream with the
  AFNO double skip. Mode-restricted weights are what FourCastNet deploys at
  720x1440, so this *is* the canonical AFNO, not a degraded clone; matching
  FNO's modes additionally makes the comparison strictly controlled.
  (A spectrum-wide variant was tried and rejected: its weights are
  resolution-dependent, which breaks checkpoint loading and the paper's
  resolution-invariance contract.)
* ``agfafno`` -- the *mechanism transplant*: identical AFNO mixer plus the
  zero-gated SDF modulation + pointwise geometry re-injection of
  ``AGFSpectralBlock``. Tests the paper's falsifiable prediction P3:
  if forgetting is caused by the band-limited periodic global channel and not
  by the FFT machinery itself, then (a) AFNO must collapse at depth like FNO,
  and (b) AGF-AFNO must be rescued exactly as AGF-NO is.

Weights are created at construction time on the mode-restricted band, so
``state_dict`` round-trips, super-resolution forwards crop/pad the band like
``SpectralConv2d``, and checkpoints load with default strict semantics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import Operator2D


class AFNOMixer(nn.Module):
    """AFNO spectral mixer for ``[B, in_ch, H, W]`` -> ``[B, out_ch, H, W]``.

    rFFT2 -> block-diagonal complex weights on the retained band ->
    complex soft-threshold shrinkage -> shared complex channel mixing ->
    iRFFT2. Frequencies outside the retained band are zeroed (band-limited
    global channel -- the property under test).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        modes_h: int,
        modes_w: int,
        sparsity_th: float = 0.001,
        chan_relaxed: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.sparsity_th = sparsity_th

        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_h, modes_w, dtype=torch.cfloat)
        )
        self.w2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_h, modes_w, dtype=torch.cfloat)
        )
        if chan_relaxed:
            self.wc = nn.Parameter(scale * torch.randn(out_ch, out_ch, dtype=torch.cfloat))
            self.bc = nn.Parameter(torch.zeros(out_ch, dtype=torch.cfloat))
        else:
            self.wc = None
            self.bc = None

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        B, C, H, W = v.shape
        mh = min(self.modes_h, H)
        mw = min(self.modes_w, W // 2 + 1)

        x_ft = torch.fft.rfft2(v, dim=(-2, -1))  # [B, C, H, W//2+1]

        pos = torch.einsum("bixy,ioxy->boxy", x_ft[..., :mh, :mw].cfloat(), self.w1[..., :mh, :mw])
        neg = torch.einsum("bixy,ioxy->boxy", x_ft[..., -mh:, :mw].cfloat(), self.w2[..., :mh, :mw])

        # Soft spectrum shrinkage: complex soft-threshold on both halves.
        pos = torch.view_as_complex(F.softshrink(torch.view_as_real(pos), self.sparsity_th))
        neg = torch.view_as_complex(F.softshrink(torch.view_as_real(neg), self.sparsity_th))

        out = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=v.device)
        out[..., :mh, :mw] = pos
        out[..., -mh:, :mw] = neg

        if self.wc is not None:
            out = torch.einsum("bcxy,oc->boxy", out, self.wc) + self.bc.view(1, -1, 1, 1)

        return torch.fft.irfft2(out, s=(H, W), dim=(-2, -1))


class _PreNormMixin:
    """LayerNorm over the channel dim of a [B, C, H, W] tensor."""

    def _pre_norm(self, v: torch.Tensor) -> torch.Tensor:
        return self.norm(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class AFNOBlock(nn.Module, _PreNormMixin):
    """Pre-norm residual block wrapping :class:`AFNOMixer` (the AFNO design:
    LayerNorm -> mixer -> residual, double skip -> LayerNorm -> MLP -> residual).

    Accepts (and ignores) ``sdf``/``coords`` so it drops into the same block
    interface every experiment suite and the probe already drive.
    """

    def __init__(
        self,
        width: int,
        modes_h: int,
        modes_w: int,
        sdf_ch: int = 0,  # unused; interface parity
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        double_skip: bool = True,
        sparsity_th: float = 0.001,
        **kwargs,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.double_skip = double_skip
        hidden = mlp_ratio * width
        self.mlp = nn.Sequential(
            nn.Conv2d(width, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, width, 1),
        )
        # Full-spectrum variant (every mode kept) would need lazily-built,
        # resolution-dependent parameters -- incompatible with the probe's
        # strict checkpoint loading and with the O(N log N) story at fixed
        # mode budget. The AFNO *as deployed* (FourCastNet, 720x1440) uses
        # mode-restricted weights, so the band-limited mixer below IS the
        # canonical architecture; the full-spectrum ablation is out of scope.
        self.mixer = AFNOMixer(width, width, modes_h, modes_w,
                               sparsity_th=sparsity_th)

    def forward(self, v: torch.Tensor, sdf: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        h = self._pre_norm(v)
        v = v + self.mixer(h)
        if self.double_skip:
            h2 = self._pre_norm(v)
            v = v + self.mlp(h2)
        else:
            v = v + self.mlp(v)
        return v


class AGFAFNOBlock(AFNOBlock, _PreNormMixin):
    """AFNO mixer + the zero-gated geometry mechanism of ``AGFSpectralBlock``.

    Exact transplant of the AGF mechanism onto the AFNO backbone:

    * ``gate_spec``-modulated SDF-derived field multiplies the mixer output,
    * ``gate_mlp`` re-injects the raw SDF (+coords) in the pointwise branch.

    Both gates start at zero, so at initialization this block is *exactly*
    ``AFNOBlock`` -- the comparison against plain AFNO is controlled.
    """

    def __init__(self, width: int, modes_h: int, modes_w: int, sdf_ch: int = 3,
                 mlp_ratio: int = 2, dropout: float = 0.0, gate_max: float = 5.0,
                 use_coord_features: bool = True, n_coord_features: int = 4,
                 sparsity_th: float = 0.001, **kwargs):
        super().__init__(width, modes_h, modes_w, sdf_ch=sdf_ch, mlp_ratio=mlp_ratio,
                         dropout=dropout, sparsity_th=sparsity_th)
        self.gate_max = gate_max
        # Geometry spectral path: SDF channels -> modulation field. Uses its
        # own mixer with in_ch = sdf_ch (out_ch = width, like AGFSpectralBlock).
        self.geo_mixer = AFNOMixer(sdf_ch, width, modes_h, modes_w,
                                   sparsity_th=sparsity_th)
        self.sdf_lin = nn.Conv2d(sdf_ch, width, 1)
        self.gate_spec = nn.Parameter(torch.zeros(1))
        extra = sdf_ch + (n_coord_features if use_coord_features else 0)
        self.mlp_geo = nn.Sequential(
            nn.Conv2d(width + extra, mlp_ratio * width, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_ratio * width, width, 1),
        )
        self.gate_mlp = nn.Parameter(torch.zeros(1))
        self.use_coord_features = use_coord_features
        self.n_coord_features = n_coord_features

    @staticmethod
    def _coords(B: int, H: int, W: int, device, n: int) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, H, device=device)
        xs = torch.linspace(0.0, 1.0, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        feats = [torch.sin(2 * math.pi * gx), torch.cos(2 * math.pi * gx),
                 torch.sin(2 * math.pi * gy), torch.cos(2 * math.pi * gy)][:n]
        return torch.stack(feats, dim=0)[None].expand(B, -1, -1, -1)

    def forward(self, v: torch.Tensor, sdf: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        B, C, H, W = v.shape
        # --- primary stream: pre-norm + AFNO mixer ------------------------
        h = self._pre_norm(v)
        mixed = self.mixer(h)

        # --- zero-gated geometric modulation of the mixer output ----------
        g_field = self.geo_mixer(sdf) + self.sdf_lin(sdf)
        gs = self.gate_spec.clamp(-self.gate_max, self.gate_max)
        v = v + mixed * (1.0 + gs * torch.tanh(g_field))

        # --- pointwise branch + gated geometry re-injection ---------------
        h2 = self._pre_norm(v)
        out = v + self.mlp(h2)
        if self.use_coord_features:
            if coords is None:
                coords = self._coords(B, H, W, v.device, self.n_coord_features)
            h_in = torch.cat([h2, sdf, coords], dim=1)
        else:
            h_in = torch.cat([h2, sdf], dim=1)
        gm = self.gate_mlp.clamp(-self.gate_max, self.gate_max)
        return out + gm * self.mlp_geo(h_in)


def afno2d(cfg) -> Operator2D:
    """AFNO backbone on the shared skeleton (mode-restricted, canonical)."""
    return Operator2D(
        kind="afno",
        in_ch=cfg.in_ch,
        out_ch=cfg.out_ch,
        width=cfg.width,
        n_blocks=cfg.n_blocks,
        modes_h=cfg.modes_h,
        modes_w=cfg.modes_w,
        sdf_ch=cfg.sdf_ch,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        block_cls=AFNOBlock,
        block_kwargs={"sparsity_th": 0.001},
    )


def agfafno2d(cfg, gate_mode: str = "full") -> Operator2D:
    """AGF-NO mechanism transplanted onto the AFNO backbone."""
    return Operator2D(
        kind="agfafno",
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
        block_cls=AGFAFNOBlock,
        block_kwargs={"sparsity_th": 0.001},
    )
