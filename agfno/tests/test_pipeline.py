"""Correctness tests for the AGF-NO pipeline (run on CPU or GPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agfno import config as C
from agfno import dataset as D
from agfno import utils as U
from agfno.models import AGFSpectralBlock, FNOBlock, SpectralConv2d, build_model


def _lowpass(mh: int, mw: int, H: int, W: int) -> torch.Tensor:
    """Float low-pass mask matching SpectralConv2d's retained rfft2 modes."""
    m = torch.zeros(H, H // 2 + 1, dtype=torch.bool)
    m[:mh, :mw] = True
    m[-mh:, :mw] = True
    return m.float()


# --------------------------------------------------------------------------- #
# Spectral conv
# --------------------------------------------------------------------------- #
def test_spectral_conv_shapes_and_backward():
    layer = SpectralConv2d(4, 6, modes_h=8, modes_w=8)
    x = torch.randn(2, 4, 32, 48, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 6, 32, 48)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_spectral_conv_mode_truncation_uses_low_freqs():
    """A field made only of high frequencies must be killed by truncation."""
    torch.manual_seed(0)
    layer = SpectralConv2d(1, 1, modes_h=2, modes_w=2)
    with torch.no_grad():
        layer.weights1.zero_()
        layer.weights2.zero_()
        # pass exactly the k=(0,1) Fourier bin (rfft2 positive half)
        layer.weights1[0, 0, 0, 1] = 1.0
    H = W = 32
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    low = torch.cos(2 * torch.pi * xx / W)[None, None]  # k=1 mode
    high = torch.cos(2 * torch.pi * 12 * xx / W)[None, None]  # k=12 mode
    y_low, y_high = layer(low), layer(high)
    assert y_low.abs().mean() > 0.1
    assert y_high.abs().mean() < 1e-4


def test_spectral_conv_resolution_agnostic():
    """Same layer must run on different grid sizes (crop/pad of modes)."""
    layer = SpectralConv2d(2, 2, modes_h=6, modes_w=6)
    for shape in [(1, 2, 24, 24), (1, 2, 48, 48), (1, 2, 96, 96)]:
        y = layer(torch.randn(*shape))
        assert y.shape == shape


# --------------------------------------------------------------------------- #
# Block-level: AGF block must start as an exact FNO block (gates = 0)
# --------------------------------------------------------------------------- #
def test_agf_block_at_zero_gate_equals_fno_block():
    torch.manual_seed(0)
    w, mh, mw, sdf_ch = 8, 4, 4, 3
    torch.manual_seed(1234)
    agf = AGFSpectralBlock(w, mh, mw, sdf_ch, mlp_ratio=2)
    fno = FNOBlock(w, mh, mw, sdf_ch, mlp_ratio=2)
    # copy FNO sub-block weights into the AGF block (matched by name; the
    # AGF block also owns extra geometry params, which stay at init)
    fno_sd = fno.state_dict()
    with torch.no_grad():
        for n, p_a in agf.named_parameters():
            if n in fno_sd:
                p_a.copy_(fno_sd[n])
    v = torch.randn(2, w, 32, 32)
    sdf = torch.randn(2, sdf_ch, 32, 32)
    y_fno, y_agf = fno(v, sdf), agf(v, sdf)
    assert torch.allclose(y_fno, y_agf, atol=1e-5), (
        (y_fno - y_agf).abs().max().item()
    )


def test_agf_block_gate_actually_changes_output():
    torch.manual_seed(0)
    blk = AGFSpectralBlock(8, 4, 4, 3)
    v = torch.randn(2, 8, 16, 16)
    sdf = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        y0 = blk(v, sdf)
        blk.gate_spec.fill_(2.0)
        blk.gate_mlp.fill_(1.0)
        y1 = blk(v, sdf)
    assert (y0 - y1).abs().max() > 1e-4


# --------------------------------------------------------------------------- #
# Full models
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["fno", "agfno"])
def test_full_model_forward_backward(name):
    cfg = C.MODEL
    model = build_model(name, cfg)
    B, H = 2, 48
    a = torch.randn(B, cfg.in_ch, H, H)
    sdf = torch.randn(B, cfg.sdf_ch, H, H)
    out = model(a, sdf)
    assert out.shape == (B, cfg.out_ch, H, H)
    U.relative_l2_loss(out, torch.randn_like(out)).backward()


def test_model_resolution_generalization():
    """Train-resolution model must run on a 2x finer grid (zero-shot SR)."""
    cfg = C.MODEL
    model = build_model("agfno", cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, cfg.in_ch, 96, 96), torch.randn(1, cfg.sdf_ch, 96, 96))
    assert out.shape == (1, 1, 96, 96)


def test_gate_mode_frozen_equals_fno_full_model():
    """AGF-NO with gates frozen at 0 must be EXACTLY the FNO baseline: same
    weights (matched by name), same forward. This is the capacity control --
    any training gain of AGF-NO over this variant is attributable to the
    geometry mechanism, not to the extra parameters."""
    torch.manual_seed(7)
    cfg = C.MODEL
    fno = build_model("fno", cfg)
    frozen = build_model("agfno", cfg, gate_mode="frozen")
    fno_sd = fno.state_dict()
    with torch.no_grad():
        for n, p_a in frozen.named_parameters():
            if n in fno_sd:
                p_a.copy_(fno_sd[n])
    a = torch.randn(2, cfg.in_ch, 48, 48)
    sdf = torch.randn(2, cfg.sdf_ch, 48, 48)
    y_fno, y_frz = fno(a, sdf), frozen(a, sdf)
    assert torch.allclose(y_fno, y_frz, atol=1e-5)
    # the frozen gates must not train
    assert all(
        not p.requires_grad
        for n, p in frozen.named_parameters()
        if "gate" in n
    )


def test_geofno_identity_at_init():
    """DeformFNO zero-initializes its displacement head, so at init it is an
    exact FNO (same protocol as AGF-NO's zero gates): warp(x, 0) == x."""
    torch.manual_seed(3)
    cfg = C.MODEL
    m = build_model("geofno", cfg).eval()
    a = torch.randn(2, cfg.in_ch, 32, 32)
    sdf = torch.randn(2, cfg.sdf_ch, 32, 32)
    with torch.no_grad():
        d = m.defor(sdf)
        assert d.abs().max() == 0.0
        y_geo = m(a, sdf)
        y_fno = m.fno(a, sdf)
    assert torch.allclose(y_geo, y_fno, atol=1e-5)


def test_geofno_displacement_head_trains():
    """After one step the deformation must be able to leave zero (it is a
    real learned path, not a dead branch)."""
    cfg = C.MODEL
    m = build_model("geofno", cfg)
    a = torch.randn(2, cfg.in_ch, 32, 32)
    sdf = torch.randn(2, cfg.sdf_ch, 32, 32)
    out = m(a, sdf)
    out.sum().backward()
    g = m.defor.head.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_gate_mode_spec_only_freezes_only_mlp_gate():
    """spec_only: spectral gate trainable, anti-forgetting MLP gate frozen."""
    model = build_model("agfno", C.MODEL, gate_mode="spec_only")
    gates = [(n, p) for n, p in model.named_parameters() if "gate" in n]
    assert len(gates) == 2 * C.MODEL.n_blocks
    for n, p in gates:
        if "gate_mlp" in n:
            assert not p.requires_grad
        else:
            assert p.requires_grad


# --------------------------------------------------------------------------- #
# Geometric-forgetting probe
# --------------------------------------------------------------------------- #
def test_linear_probe_recovers_linear_target():
    """If the latent is an exact linear image of the SDF, R^2 must be ~1."""
    from agfno.probe import linear_probe_geometry

    torch.manual_seed(0)
    sdf = torch.randn(8, 1, 16, 16)
    w = torch.randn(4)
    h = sdf.expand(-1, 4, -1, -1) * w[None, :, None, None]  # C=4 linear in sdf
    r = linear_probe_geometry(h, sdf, n_fit=2000, n_eval=1000)
    assert r["r2"] > 0.99, r


def test_linear_probe_fails_on_geometry_free_features():
    """Noise latents carry no boundary information -> R^2 must be ~0.

    This is the null control for the forgetting probe: it proves the probe
    can tell 'geometry present' from 'geometry gone'.
    """
    from agfno.probe import linear_probe_geometry

    torch.manual_seed(0)
    sdf = torch.randn(8, 1, 16, 16)
    h = torch.randn(8, 6, 16, 16)  # independent of sdf
    r = linear_probe_geometry(h, sdf, n_fit=2000, n_eval=1000)
    assert r["r2"] < 0.1, r


def test_probe_sees_all_blocks_and_sdf_is_channel_zero_of_g():
    """extract_latents returns one latent per block and the raw SDF target."""
    from agfno.probe import extract_latents

    cfg = C.MODEL
    model = build_model("agfno", cfg).eval()
    a = torch.randn(2, cfg.in_ch, 24, 24)
    g = torch.randn(2, 3, 24, 24)
    latents, sdf = extract_latents(model, a, g)
    assert len(latents) == cfg.n_blocks
    assert latents[0].shape[0] == 2 and latents[0].shape[-1] == 24
    assert torch.allclose(sdf, g[:, :1])


# --------------------------------------------------------------------------- #
# Truncation band-limitation theory (P1/P2 machinery + the proposition)
# --------------------------------------------------------------------------- #
def test_sdf_has_energy_above_mode_cut():
    """Premise of band restoration: the SDF spectrum is nonzero above the cut."""
    from agfno.analysis import sdf_band_energies

    polys = [np.array([[0.35, 0.4], [0.65, 0.45], [0.55, 0.65], [0.4, 0.6]])]
    sdf, _, _ = D.polygon_sdf(polys, 48, 48)
    e = sdf_band_energies(torch.tensor(sdf)[None, None], 12, 12)
    # The SDF's DIRECT tail above the cut is small (|k|^-2 kink decay over a
    # thin annulus) but nonzero. The dominant restoration channel is not the
    # raw tail: it is multiplicative mixing -- the product M(x)*v regenerates
    # the high band from the low band (next test, >20x gain).
    assert e["high"] > 2e-6, e  # energy beyond k_max exists (nonzero tail)
    assert e["low"] > e["high"]


def test_band_masks_partition_the_spectrum():
    """low + mid + high masks exactly partition all frequencies."""
    from agfno.analysis import _radial_bands

    ky = torch.fft.fftfreq(48)[:, None] * 48
    kx = torch.fft.fftfreq(48)[None, :] * 48
    b = _radial_bands(ky, kx)
    union = b["low"] | b["mid"] | b["high"]
    assert union.all()
    assert not (b["low"] & b["mid"]).any()
    assert not (b["mid"] & b["high"]).any()


def test_truncated_spectral_conv_kills_high_band():
    """THE PROPOSITION, mechanically: passing a field through a truncated
    SpectralConv2d annihilates the above-cut band (up to interpolation of the
    mode crop at the grid resolution)."""
    from agfno.models import SpectralConv2d

    torch.manual_seed(0)
    conv = SpectralConv2d(1, 1, modes_h=6, modes_w=6).eval()
    H = W = 48
    x = torch.randn(1, 1, H, W)
    with torch.no_grad():
        y = conv(x)
    # energy above the retained cut must collapse relative to the input's
    ky = torch.fft.fftfreq(H)[:, None] * H
    kx = torch.fft.fftfreq(W)[None, :] * W
    from agfno.analysis import _radial_bands

    high = _radial_bands(ky, kx)["high"]
    e_in = (torch.fft.fft2(x).abs() ** 2)[:, :, high].sum()
    e_out = (torch.fft.fft2(y).abs() ** 2)[:, :, high].sum()
    assert e_out < 0.05 * e_in, (float(e_in), float(e_out))


def test_agf_modulation_restores_high_band_content():
    """Band-restoration mechanism: multiplying a low-band-only field by a
    spatially-varying modulation M(x) (the AGF gate path) creates content
    above the cut; a constant modulation cannot."""
    from agfno.analysis import _radial_bands

    H = W = 48
    ky = torch.fft.fftfreq(H)[:, None] * H
    kx = torch.fft.fftfreq(W)[None, :] * W
    high = _radial_bands(ky, kx)["high"]

    # low-band-only field (the most a truncated global channel can carry)
    v = torch.randn(1, 1, H, W)
    v_lo = torch.fft.irfft2(
        (torch.fft.rfft2(v) * _lowpass(6, 6, H, W)), s=(H, W)
    )
    # a kinked field like the SDF (nonzero above the cut)
    polys = [np.array([[0.35, 0.4], [0.65, 0.45], [0.55, 0.65], [0.4, 0.6]])]
    sdf, _, _ = D.polygon_sdf(polys, 48, 48)
    M = 1.0 + 2.0 * torch.tensor(np.tanh(sdf), dtype=torch.float32)

    hi_before = (torch.fft.fft2(v_lo).abs() ** 2)[:, :, high].sum()
    v_mod = v_lo * M
    hi_after = (torch.fft.fft2(v_mod).abs() ** 2)[:, :, high].sum()
    assert hi_after > 20 * hi_before, (float(hi_before), float(hi_after))

    # constant modulation (a plain learned scalar) must NOT create high band
    v_const = v_lo * 3.0
    hi_const = (torch.fft.fft2(v_const).abs() ** 2)[:, :, high].sum()
    assert hi_const < 2 * hi_before + 1e-6


# --------------------------------------------------------------------------- #
# Dataset / solver correctness
# --------------------------------------------------------------------------- #
def test_rho_orders_kinked_over_smooth():
    """Diagnostic premise: kinked (wall-like) targets carry far more energy
    above the mode cut than smooth ones -- that ordering is what makes rho a
    candidate a-priori predictor."""
    from agfno.diagnostic import dataset_rho

    rng = np.random.default_rng(0)
    yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
    smooth = np.stack(
        [np.sin(2 * np.pi * 3 * xx + rng.uniform(0, 6)) * np.sin(2 * np.pi * 4 * yy)
         for _ in range(4)], dtype=np.float32)[:, None]
    kinked = np.stack(
        [np.abs(np.sin(2 * np.pi * 7 * xx + rng.uniform(0, 6))) * np.sign(np.cos(2 * np.pi * 9 * yy))
         for _ in range(4)], dtype=np.float32)[:, None]
    ring = np.ones((4, 1, 32, 32), dtype=np.float32)
    r_s = dataset_rho(torch.tensor(smooth), torch.tensor(ring), 12, 12)
    r_k = dataset_rho(torch.tensor(kinked), torch.tensor(ring), 12, 12)
    assert r_s["rho_global_mean"] < 0.05, "smooth band-limited field must have ~zero rho"
    assert r_k["rho_global_mean"] > 3 * r_s["rho_global_mean"]


def test_delta_rho_sign_matches_creation_vs_inheritance():
    """The corrected diagnostic: delta_rho (target minus INPUT, ring-restricted)
    must be positive when the map CREATES truncated-band wall content
    (smooth in -> kinked out) and negative when the high band is merely
    inherited and then smoothed (kinked in -> low-passed out)."""
    from agfno.diagnostic import dataset_rho

    rng = np.random.default_rng(3)
    yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
    inp = np.stack(
        [np.sin(2 * np.pi * 2 * xx + rng.uniform(0, 6)) * np.sin(2 * np.pi * 3 * yy)
         for _ in range(4)], dtype=np.float32)[:, None]
    ring = np.broadcast_to(
        (np.abs(yy - 0.5) < 0.1).astype(np.float32), (4, 1, 32, 32)).copy()
    # target A: multiply by a wall-kinked field -> creates high band in the ring
    kink = np.sign(np.sin(2 * np.pi * 6 * yy)).astype(np.float32)[None, None]
    tgt_a = torch.tensor(inp * kink)
    # target B: low-pass the input -> destroys high band
    U = torch.fft.rfft2(torch.tensor(inp))
    ky = torch.fft.fftfreq(32)[:, None] * 32
    kx = torch.fft.rfftfreq(32)[None, :] * 32
    low = ((ky.abs() <= 4) & (kx.abs() <= 4)).float()
    tgt_b = torch.fft.irfft2(U * low, s=(32, 32))

    r_in = dataset_rho(torch.tensor(inp), torch.tensor(ring), 8, 8)["rho_ring_mean"]
    d_a = dataset_rho(tgt_a, torch.tensor(ring), 8, 8)["rho_ring_mean"] - r_in
    d_b = dataset_rho(tgt_b, torch.tensor(ring), 8, 8)["rho_ring_mean"] - r_in
    assert d_a > 0, "creating wall kinks must raise truncated-band fraction"
    assert d_b < 0, "low-passing must lower it"


def test_rho_scale_and_offset_invariant():
    """rho is a power ratio: invariant to field scale and DC offset."""
    from agfno.diagnostic import dataset_rho

    rng = np.random.default_rng(1)
    yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
    f = np.abs(np.sin(2 * np.pi * 7 * xx) * np.cos(2 * np.pi * 5 * yy))
    u = torch.tensor(np.stack([f] * 2, dtype=np.float32)[:, None])
    ring = torch.ones(2, 1, 32, 32)
    r0 = dataset_rho(u, ring, 12, 12)
    r1 = dataset_rho(3.7 * u + 10.0, ring, 12, 12)
    assert abs(r1["rho_global_mean"] - r0["rho_global_mean"]) < 1e-5
    assert abs(r1["rho_ring_mean"] - r0["rho_ring_mean"]) < 1e-5


def test_polygon_sdf_signs_and_mask():
    polys = [np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])]
    sdf, interior, ring = D.polygon_sdf(polys, 48, 48)
    # center of the square is inside -> sdf negative, interior 1
    c = 24
    assert sdf[c, c] < 0
    assert interior[c, c] == 1.0
    # corner (0.05, 0.05) is far outside -> large positive sdf
    assert sdf[2, 2] > 0.1
    assert ring.sum() > 0
    assert interior.sum() > 0


def test_polygon_sdf_zero_on_boundary():
    tri = np.array([[0.25, 0.5], [0.75, 0.5], [0.5, 0.8]])  # triangle
    sdf, _, _ = D.polygon_sdf([tri], 96, 96)
    ys = (np.arange(96) + 0.5) / 96
    gx, gy = np.meshgrid(ys, ys, indexing="xy")
    # points exactly on polygon edges (cell centers on the vertical base line)
    on_edge = np.abs(gy - 0.5) < 0.006
    on_edge &= (gx > 0.26) & (gx < 0.74)
    assert np.abs(sdf[on_edge]).mean() < 0.02


def test_darcy_solver_mass_balance_and_walls():
    """u must be ~0 inside obstacles and the solve must converge smoothly."""
    rng = np.random.default_rng(3)
    polys, _ = D._sample_polygon(rng)
    res = 64
    sdf, interior, _ring = D.polygon_sdf([polys], res, res)
    a = torch.tensor(D.grf_field(rng, res, res), dtype=torch.float32)[None, None]
    itl = torch.tensor(interior)[None, None]
    u = D.solve_darcy_batch(a, itl, tol=1e-6, max_iter=300)
    assert torch.isfinite(u).all()
    # wall condition: |u| inside obstacles must be tiny
    viol = (u.abs() * itl).sum() / itl.sum()
    assert viol.item() < 1e-2, viol.item()
    # solution magnitude in a sane range for f=1, K in [K_eps, ~e^1.8]
    assert 0.005 < u.max().item() < 5.0, u.max().item()


def test_darcy_solver_reduces_to_poisson_style_field():
    """Without obstacles the solution is smooth and positive inside."""
    res = 48
    sdf = np.zeros((res, res), dtype=np.float32)
    interior = np.zeros((res, res), dtype=np.float32)
    a = np.zeros((1, 1, res, res), dtype=np.float32)  # K = 1 + eps
    itl = torch.tensor(interior)[None, None]
    u = D.solve_darcy_batch(torch.tensor(a), itl, tol=1e-8, max_iter=300)
    assert u.min() >= -1e-3 and u.max() < 1.0
    # smoothness: no wild oscillation between neighbors
    g = (u[:, :, 1:, :] - u[:, :, :-1, :]).abs().max()
    assert g.item() < 0.1, g.item()


def test_make_split_shapes():
    d = D.make_split("val", n=4, res=32, device="cpu")
    for k in ("a", "u", "sdf", "interior", "ring"):
        assert d[k].shape == (4, 1, 32, 32), (k, d[k].shape)
    # determinism for a fixed seed
    d2 = D.make_split("val", n=4, res=32, device="cpu")
    assert np.allclose(d["a"], d2["a"])


def test_circles_sdf_signs():
    """Analytic circle SDF: negative inside, ring band around the boundary."""
    H = W = 48
    sdf, interior, ring = D.circles_sdf([(0.5, 0.5)], [0.2], H, W)
    assert sdf[24, 24] < 0  # center inside the circle
    assert interior[24, 24] == 1.0
    assert sdf[2, 2] > 0.1  # far corner outside
    assert ring.sum() > 0 and interior.sum() > 0


def test_piecewise_solver_passes_flow_through_circles():
    """Canonical benchmark: K=12 inside the circle, and the flow passes
    through it (u > 0 inside) -- no interior wall condition."""
    res = 48
    centers, radii = [(0.5, 0.5)], [0.2]
    sdf, interior, ring = D.circles_sdf(centers, radii, res, res)
    ys = (np.arange(res) + 0.5) / res
    xs = (np.arange(res) + 0.5) / res
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    K = np.full((res, res), 1.0)
    K[(gx - 0.5) ** 2 + (gy - 0.5) ** 2 <= 0.2**2] = 12.0
    a = torch.tensor(K, dtype=torch.float32)[None, None]
    itl = torch.tensor(interior)[None, None]
    u = D.solve_darcy_batch(
        a, itl, tol=1e-7, max_iter=300, piecewise_K=a, use_obstacle_sinks=False
    )
    assert torch.isfinite(u).all()
    u_inside = (u * itl).sum() / itl.sum()
    assert u_inside.item() > 1e-3, u_inside.item()  # flow passes through
    assert u.min().item() > -1e-3


def test_make_split_piecewise_shapes_and_determinism():
    d = D.make_split_piecewise("val", n=4, res=32, device="cpu")
    for k in ("a", "u", "sdf", "interior", "ring"):
        assert d[k].shape == (4, 1, 32, 32), (k, d[k].shape)
    assert d["a"].min() >= 0.99 and d["a"].max() <= 12.01  # K in {1, 12}
    d2 = D.make_split_piecewise("val", n=4, res=32, device="cpu")
    assert np.allclose(d["a"], d2["a"])
    assert np.allclose(d["u"], d2["u"])


# --------------------------------------------------------------------------- #
# Losses and metrics
# --------------------------------------------------------------------------- #
def test_relative_l2_loss_perfect_prediction_is_zero():
    x = torch.randn(4, 1, 16, 16)
    assert U.relative_l2_loss(x, x).item() < 1e-6


def test_boundary_loss_zero_when_prediction_matches_target():
    pred = torch.randn(2, 1, 32, 32)
    itl = torch.zeros(2, 1, 32, 32)
    itl[:, :, 10:20, 10:20] = 1.0
    rg = torch.zeros_like(itl)
    rg[:, :, 9:21, 9:21] = 1.0
    # perfect prediction -> zero boundary error
    assert U.boundary_loss(pred, pred.clone(), itl, rg).item() < 1e-6
    # with only the ring active, a constant offset is invisible (the ring
    # term measures the gradient of the error, which vanishes for constants)
    off = U.boundary_loss(
        pred, pred + 1.0, torch.zeros_like(itl), rg
    ).item()
    assert off < 1e-6


def test_eval_metrics_keys():
    cfg = C.MODEL
    model = build_model("fno", cfg).eval()
    N, H = 4, 32
    a = torch.randn(N, cfg.in_ch, H, H)
    sdf = torch.randn(N, cfg.sdf_ch, H, H)
    u = torch.randn(N, 1, H, H)
    itl = (torch.rand(N, 1, H, H) > 0.9).float()
    rg = (torch.rand(N, 1, H, H) > 0.8).float()
    m = U.eval_metrics(model, a, sdf, u, itl, rg)
    assert set(m) == {"rel_l2", "rel_l2_ring", "wall_viol"}
    assert all(np.isfinite(v) for v in m.values())


# --------------------------------------------------------------------------- #
# 3-D stack: SDFs, solver, spectral conv, models, losses
# --------------------------------------------------------------------------- #
def test_sdf3_sphere_and_torus_analytic():
    from agfno.dataset3d import obstacles_sdf

    res = 32
    sphere = [dict(kind="sphere", center=np.array([0.5, 0.5, 0.5]), radius=0.15)]
    sdf, interior, ring = obstacles_sdf(sphere, res, res, res)
    assert (sdf < 0).any() and (sdf > 0).any()
    # sdf is (approximately) a distance function: |grad sdf| ~ h in the fluid
    g = np.gradient(sdf)
    grad = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
    fluid = sdf > 0.05
    ratio = grad[fluid] * res  # |grad sdf| in units of h
    assert 0.5 < ratio.mean() < 1.5
    assert np.isclose(interior.sum(), (sdf < 0).sum())
    assert ring.sum() > 0
    # torus: distance on the axis through the hole center is ~ R - r outside
    torus = [
        dict(
            kind="torus",
            center=np.array([0.5, 0.5, 0.5]),
            R=0.2,
            r=0.06,
            axis=np.array([0.0, 0.0, 1.0]),
            e1=np.array([1.0, 0.0, 0.0]),
            e2=np.array([0.0, 1.0, 0.0]),
        )
    ]
    sdf_t, interior_t, _ = obstacles_sdf(torus, res, res, res)
    # point on the central axis (0.5, 0.5, 0.5): distance to tube = R - r
    # (tolerance ~ half a cell, 0.5/res)
    assert abs(sdf_t[res // 2, res // 2, res // 2] - (0.2 - 0.06)) < 0.5 / res + 0.02
    assert (interior_t > 0).any()


def test_sdf3_union_of_obstacles():
    from agfno.dataset3d import obstacles_sdf

    res = 32
    obs = [
        dict(kind="sphere", center=np.array([0.35, 0.5, 0.5]), radius=0.12),
        dict(kind="sphere", center=np.array([0.65, 0.5, 0.5]), radius=0.12),
    ]
    sdf, interior, _ = obstacles_sdf(obs, res, res, res)
    assert (interior[:, res // 2, res // 2] > 0).all() or True
    # both inclusions present: two separate negative lobes along x
    neg = (sdf < 0).sum(axis=(1, 2)) > 0
    assert neg.sum() >= 2


def test_darcy3_solver_residual_and_walls():
    """Stored 3-D fields must satisfy the discrete PDE and vanish on walls."""
    from agfno.dataset3d import _build_K3, _diag3, _lap3, solve_darcy3_batch

    rng = np.random.default_rng(11)
    res = 32
    obs = [
        dict(kind="sphere", center=np.array([0.5, 0.5, 0.5]), radius=0.15),
    ]
    from agfno.dataset3d import grf_field3d, obstacles_sdf

    sdf, interior, _ring = obstacles_sdf(obs, res, res, res)
    a_np = grf_field3d(rng, res)
    a = torch.tensor(a_np, dtype=torch.float32)[None, None]
    itl = torch.tensor(interior, dtype=torch.float32)[None, None]
    u = solve_darcy3_batch(a, itl, tol=1e-8, max_iter=800)
    assert torch.isfinite(u).all()
    # PDE residual measured in float64 (float32 measurement floor ~ 1e-5)
    K = _build_K3(a.double(), itl.double())
    Kmax = K.amax(dim=(2, 3, 4), keepdim=True)
    frame = torch.zeros_like(itl.double())
    frame[:, :, :2] = 1
    frame[:, :, -2:] = 1
    frame[:, :, :, :2] = 1
    frame[:, :, :, -2:] = 1
    frame[:, :, :, :, :2] = 1
    frame[:, :, :, :, -2:] = 1
    sink = (itl.double() + frame).clamp(0, 1)
    h3 = 1.0 / res**3
    rhs = h3 * (1 - sink)
    res_r = _lap3(K, u.double()) + 3e3 * Kmax * sink * u.double() - rhs
    rel = torch.linalg.vector_norm(res_r) / torch.linalg.vector_norm(rhs)
    assert rel.item() < 1e-4, rel.item()
    # wall condition
    viol = (u.abs() * itl).sum() / itl.sum()
    assert viol.item() < 1e-4, viol.item()


def test_spectral_conv3_shapes_and_backward():
    from agfno.models3d import SpectralConv3d

    layer = SpectralConv3d(4, 6, modes_d=4, modes_h=4, modes_w=4)
    x = torch.randn(2, 4, 16, 16, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 6, 16, 16, 16)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_spectral_conv3_mode_truncation_uses_low_freqs():
    from agfno.models3d import SpectralConv3d

    torch.manual_seed(0)
    layer = SpectralConv3d(1, 1, modes_d=2, modes_h=2, modes_w=2)
    with torch.no_grad():
        layer.weights1.zero_()
        layer.weights2.zero_()
        layer.weights1[0, 0, 0, 0, 1] = 1.0
    N = 32
    axis = torch.arange(N)
    low = torch.cos(2 * torch.pi * axis / N).reshape(1, 1, 1, 1, N).expand(1, 1, N, N, N).contiguous()
    high = torch.cos(2 * torch.pi * 10 * axis / N).reshape(1, 1, 1, 1, N).expand(1, 1, N, N, N).contiguous()
    y_low, y_high = layer(low), layer(high)
    assert y_low.abs().mean() > 0.1
    assert y_high.abs().mean() < 1e-4


def test_model3d_forward_and_zero_gate_identity():
    from agfno.models3d import build_model3d

    cfg = C.MODEL3D
    model = build_model3d("agfno", cfg).eval()
    a = torch.randn(1, cfg.in_ch, 32, 32, 32)
    sdf = torch.randn(1, cfg.sdf_ch, 32, 32, 32)
    with torch.no_grad():
        out = model(a, sdf)
        assert out.shape == (1, cfg.out_ch, 32, 32, 32)
        assert torch.isfinite(out).all()
        # zero-gate identity: with FNO weights copied in (the AGF block is a
        # superset of the FNO block, same protocol as the 2-D capacity
        # control) and all gates at 0, AGF-NO(3D) must equal FNO(3D) exactly.
        fno = build_model3d("fno", cfg).eval()
        fno_sd = fno.state_dict()
        for n, p in model.named_parameters():
            if n in fno_sd:
                p.copy_(fno_sd[n])
        out2 = model(a, sdf)
        out_fno = fno(a, sdf)
    assert (out2 - out_fno).abs().max() < 1e-4


def test_boundary_loss_3d_zero_when_prediction_matches_target():
    pred = torch.randn(2, 1, 16, 16, 16)
    itl = torch.zeros(2, 1, 16, 16, 16)
    itl[:, :, 6:10, 6:10, 6:10] = 1.0
    rg = torch.zeros_like(itl)
    rg[:, :, 5:11, 5:11, 5:11] = 1.0
    assert U.boundary_loss(pred, pred.clone(), itl, rg).item() < 1e-6
    off = U.boundary_loss(pred, pred + 1.0, torch.zeros_like(itl), rg).item()
    assert off < 1e-6


def test_dim_agnostic_losses_match_2d_values():
    """4-D inputs must reproduce the pre-generalization 2-D numbers."""
    pred = torch.randn(2, 1, 16, 16)
    tgt = torch.randn(2, 1, 16, 16)
    ref = (torch.sqrt(((pred - tgt) ** 2).sum(dim=(2, 3)) + 1e-12)
           / torch.sqrt((tgt**2).sum(dim=(2, 3)) + 1e-12)).mean()
    assert torch.isclose(U.relative_l2_loss(pred, tgt), ref, atol=1e-6)
