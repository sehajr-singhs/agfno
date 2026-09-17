# Theory: Truncation Band-Limitation and Its Restoration

The novelty claim of AGF-NO, stated sharply:

> Geometric forgetting in FNOs is not only a Markovian-mixing phenomenon —
> it is a **frequency-domain phenomenon**. Mode truncation band-limits the
> operator's *global* channel, so every long-range boundary interaction and
> every sharp-boundary component is structurally absent from the global
> mixing path. AGF-NO is not a decoration on the FNO block; it is the
> minimal operator that **restores the truncated band** while keeping
> O(N log N) cost.

Diagnosis → proposition → three falsifiable predictions → measurements.

---

## 1. Setup

An FNO block computes, with `v` the latent field and `W` the learnable complex
spectral weight (per retained mode pair):

```
(S v)(x) = IFFT[ W(k) · FFT[v](k) ]      for |k_i| ≤ k_max
         = 0                              otherwise
```

The output `y = σ( S v + P v )` adds a pointwise linear branch `P` (1×1 conv)
and a pointwise MLP. **The only global (all-to-all) mixing channel in the block
is `S`, and it is band-limited to `|k| ≤ k_max` by construction.**

`P` and the MLP act pointwise: they mix *channels* at a pixel, never *pixels*.
Information can therefore travel long range **only through `S`**, and anything
outside the retained band is annihilated there.

## 2. Proposition (truncation band-limitation)

Let `u` be the ground-truth solution field and `b` any boundary-induced
component of `u` (wall discontinuities, boundary layers). Write the 2-D
spatial-frequency split of any field `f` as

```
f = f_lo + f_hi,    f_lo = P_{|k|≤k_max} f,   f_hi = (I − P) f.
```

**Proposition.** After one FNO block, the global channel contributes
`S v = P v` — exactly the low-band projection. Consequently:

(a) *High-band boundary content cannot be created by the global channel.*
    The residual high-frequency content of the latent state can only come
    from pointwise nonlinearities acting on low-band inputs (spectral
    folding / harmonic generation), which is data-dependent and cannot
    represent an arbitrary sharp boundary field at the correct location and
    phase.

(b) *Every long-range boundary interaction is low-pass filtered.*
    A wall perturbation at position `x_w` reaching a query point `x_q` must
    pass through `S`; the reachability kernel is therefore a DIRICHLET
    (sinc) kernel of bandwidth `k_max`, not the Green's function of the PDE.
    Boundary interactions requiring content above the cut are transmitted
    with the wrong shape.

(c) *Depth compounds the loss.* Composing L blocks, the reachable global
    band of the composition is still `|k| ≤ k_max` per block *and* the
    effective aperture shrinks in practice: products of spectral weights
    concentrate on the modes with the largest learned gains, so decodable
    boundary content decays with depth — the measurable form of geometric
    forgetting (tested in Prediction 3).

*Proof sketch.* (a) and (b) follow from linearity of `S` and the projection
identity `S = S ∘ P`; (c) is an empirical statement — the "for all practical
weights" qualifier is exactly what the probe tests. Full elementwise proof of
(a)–(b) is two lines; we claim no more. The value is that the band-limitation
is *structural*, not statistical. ∎

## 3. What AGF-NO changes, in this language

The AGF block computes

```
(S v)(x) = IFFT[ W(k) · (1 + g·tanh(S_sdf(k))) · FFT[v](k) ]
```

Two things happen:

1. **Per-pixel re-localization.** Multiplication in frequency space by a
   *field* `M(x) = 1 + g·tanh(s(x))` (not a constant) corresponds in physical
   space to *convolution* with the kernel `m̂ = δ + g·tanh(s)·`. The global
   channel is no longer shift-invariant: its effective kernel can localize
   around walls, exactly where band-limited sinc kernels are wrong.
2. **Band restoration (the honest version, measured).** `s(x)` (the SDF) has
   a kink at the wall, so its spectrum decays like `|k|^{−2}` and its *direct*
   tail above the cut is nonzero but small (measured: ~10⁻⁵ of total energy
   for our obstacles — a thin annulus at high |k|). The dominant restoration
   channel is **multiplicative mixing**: a product of two fields has a
   spectrum equal to the *convolution* of their spectra, so even if `v` is
   strictly low-band, `M(x)·v` regenerates the high band from `M`'s full
   spectrum (measured: >20× high-band energy gain; a constant modulation
   creates none — both unit-tested). The next block's truncated convolution
   `S(v·M)` therefore acts on a field that carries boundary structure above
   the cut. This is how the high band survives truncation.

So: **pointwise-in-frequency, field-valued modulation = spatially-adaptive
kernel = band restoration.** This is the precise sense in which AGF-NO is the
"restore the truncated band" fix, not an ornament.

## 4. Three falsifiable predictions

**P1 (propagation test).** A wall perturbation propagates *further and with
the correct shape* in AGF-NO's latent than in FNO's latent: band-limitation
confines FNO's reachable influence to the sinc kernel of width ~1/k_max, the
modulation escapes it.
*Measured by:* `frequency_resolved_error` in `agfno/analysis.py` (per-band
error on `perturb_vs_clean` pairs: corrupt a clean field inside an obstacle,
propagate through the trained operator's latent, measure where the damage
shows up) and by the band-resolved spectrum of `s` (Prediction P3's premise).

**P2 (spectral signature).** The high-band spectrum of the *error* concentrates
near walls for FNO but is reduced for AGF-NO at matched budget: truncation
cannot carry the wall kink, so FNO's misfit must pile up at high |k| in the
ring, while AGF-NO re-injects it.
*Measured by:* band-resolved ring error (radial spectrum of the error split
into low/mid/high bands × ring/interior masks) in `analysis.py`.

**P3 (probe plateau / aperture hypothesis).** If forgetting is driven by the
shrinking effective aperture (weight concentration over modes × depth), the
probe R² should **plateau** — not crash — once the geometry re-injection has
compensated the aperture loss. In the previous depth sweep (1→4) FNO fell
0.90 → 0.47 with no plateau in sight; at depths 8–16, FNO should keep falling
while AGF-NO flattens.
*Measured by:* the linear geometry probe at depths 6/8/16
(`agfno/probe.py`, checkpoints from `experiments2.py`).

Each prediction is independently falsifiable; P2 has no free parameters once
`k_max` and the band edges are fixed by the architecture.

**P4 (scope prediction, added after the PDE2 near-null).** The theorem is a
statement about what the operator must *create* above the cut, not what it
must *preserve*: high-band content inherited from the input is annihilated
in the global path exactly like content the solve produces. So mechanism
value should track the CHANGE in ring-restricted truncated-band energy from
input to target,

    delta_rho = rho_ring(target) - rho_ring(input),   
    rho_ring(f) = (energy of f * ring above k_max) / (energy of f * ring),

computable before any training. Measured (`agfno/diagnostic.py`): PDE1
(+0.0244, mechanism ring gain 0.445) vs PDE2 (-0.0236, gain 0.965) -- equal
magnitude, opposite sign, gains order with it. The naive candidate (target
rho alone) INVERTS on this pair -- PDE2's targets carry more high-band
energy because their initial conditions are pre-cut at walls -- and is
retained as a documented negative result. With two families this is a
demonstration of form, not a validated selection rule; the code computes
delta_rho for any candidate dataset in seconds.

**P5 (dimension-independence, added with the 3-D replication).** Nothing in
Sections 1–4 refers to dimension: the band-limitation is a statement about
the global channel's spectrum, the Markovian decay about operator
composition, the restoration about products of fields. The collapse/plateau
phenomenon should therefore lift to 3-D unchanged. Measured
(`agfno/experiments4.py`, res-32 sphere+torus Darcy, float64 ground truth,
three seeds): the depth-16 collapse of FNO and the AGF-NO plateau reproduce
with the probe separating behaviour from representation exactly as in 2-D.
Tori additionally keep the domain multiply-connected in 3-D, where the
deformation-based escape (Geo-FNO family) remains structurally unavailable.

---

## 5. Relation to prior practice (honesty section)

- Spectral bias of truncated operators is folklore; stating it as a
  *structural band-limitation of the global channel with a restoration
  mechanism* and then *measuring* it (P1–P3) is the contribution.
- The modulation `1 + g·tanh(S_sdf)` is deliberately the minimal
  multiplicative field: pointwise-in-frequency, zero-gated at init
  (exact-FNO start), O(N log N). We do not claim novelty for "conditioning
  on geometry" in general (Geo-FNO, GINO do this differently); the claim is
  the band-restoration identity and its measurements.
- P1's "propagation" framing is inspired by kernel/NTK analyses of
  over-smoothing in GNNs; applying the reachability-kernel view to spectral
  operators appears new.

---

## 6. Formal statements and proofs

Everything here is elementary; the point is that the mechanism claims are now
*theorems about the implemented operator*, with the code as the model. The
aperture-decay part of the story (c) above deliberately stays empirical.

### 6.1 Setting

Discretize the unit square on an $N \\times N$ grid; identify fields with
their rFFT coefficients. Write $P_{\\le}$ for the mode-truncation projection
($|k_i| \\le k_{\\max}$ in each axis, as implemented by `SpectralConv2d`),
$P_>$ for its complement, and $\\mathcal{F}$ for the DFT. A *truncated
spectral conv* is $S = \\mathcal{F}^{-1} \\, W \\, P_{\\le} \\, \\mathcal{F}$
with learnable complex $W(k)$ per retained mode. A *pointwise* map $T$
has $(Tv)(x) = \\tau\\big(v(x)\\big)$ for a fixed channel map $\\tau$
(1$\\times$1 convolutions, MLPs, GELU: all pointwise). Let $g$ be the
geometry input (SDF channels) and $z_s = C_s g + Q_s g$ the geometry
spectral-conv-plus-linear output of the AGF block (`sdf_conv` + `sdf_lin`).

The AGF block's global path is
$v_{\\mathrm{geo}} = S v \\cdot m$,
$m = 1 + g_{\\mathrm{spec}} \\cdot \\tanh(z_s)$,
$g_{\\mathrm{spec}}$ a scalar gate, matching `models.py` line-for-line.

### 6.2 Band calculus

**Lemma 1 (truncation idempotence).** $S = S \\, P_{\\le}$: the truncated
convolution cannot see the high band of its input.

*Proof.* $P_{\\le}$ is a projection, so $P_{\\le} \\, P_{\\le} = P_{\\le}$;
substituting, $S P_{\\le} v = \\mathcal{F}^{-1} W P_{\\le} P_{\\le}\\mathcal{F} v
= S v$. $\\square$

**Lemma 2 (high-band annihilation).** $\\mathrm{supp}\\, \\mathcal{F}[S v]\\subseteq
\\{|k| \\le k_{\\max}\\}$ for every $v$; equality of *reachability*: if the input
is a unit perturbation at pixel $x_w$, the global channel delivers
$K_S(x_q - x_w)$ to pixel $x_q$, where $K_S$ is the inverse transform of
$W \\cdot \\mathbf{1}_{\\{|k|\\le k_{\\max}\\}}$ — a product of Dirichlet (sinc)
factors, independent of the domain geometry.

*Proof.* Support: $\\mathcal{F}[Sv] = W P_{\\le} \\mathcal{F} v$ is supported
where $P_{\\le}$ is nonzero. Reachability: take $v = \\delta_{x_w}$ (a spike);
$\\mathcal{F}[v](k) = e^{-2\\pi i k \\cdot x_w}$, so $(S v)(x_q) =
\\mathcal{F}^{-1}[W \\mathbf{1}_{\\{|k|\\le k_{\\max}\\}} e^{-2\\pi i k\\cdot x_w}](x_q)
= K_S(x_q - x_w)$ by the shift theorem. $K_S$ depends on the learned weights
but not on the obstacle boundary $\\partial D$. $\\square$

**Lemma 3 (harmonic generation).** Let $z$ be any non-constant field with
$\\mathrm{supp}\\,\\mathcal{F}[z] \\subseteq B$. Then
$\\mathrm{supp}\\,\\mathcal{F}[\\tanh z]$ generically strictly contains $B$.

*Proof.* Pointwise, $\\tanh z = z - \\tfrac13 z^3 + O(z^5)$. The convolution
theorem gives $\\mathcal{F}[z^3] = \\mathcal{F}[z] * \\mathcal{F}[z] *
\\mathcal{F}[z]$, whose support is the Minkowski sum $B+B+B$, strictly larger
than $B$ whenever $\\mathcal{F}[z] \\neq 0$ is supported on a set with at least
two non-antipodal points. The higher-order terms only enlarge the support;
exact cancellation of the cubic band would require a measure-zero coincidence
of coefficients. $\\square$

### 6.3 Theorems

**Theorem 1 (band-limited global channel).** In a depth-$L$ FNO, every
block's global channel satisfies Lemmas 1–2. Hence (i) the high band of any
latent state can only originate in pointwise nonlinearities applied to
low-band fields (harmonic generation, data-dependent phase and magnitude),
and (ii) every long-range boundary interaction is transmitted through a
shift-invariant sinc kernel of fixed bandwidth $k_{\\max}$, regardless of the
PDE's boundary structure.

*Proof.* (ii) is Lemma 2 applied blockwise; compositions of the $S_l$ remain
band-limited since each factor is. (i): the latent is built from $S_l$, the
pointwise branches $P_l$, and elementwise nonlinearities; by Lemma 2 the
$S_l$ never emit high-band content, and pointwise maps are the only remaining
term. $\\square$

**Theorem 2 (band restoration).** Suppose $S v \\not\\equiv 0$ and the
modulation field $m = 1 + g_s\\tanh(z_s)$ has
$\\mathrm{supp}\\,\\mathcal{F}[m] \\not\\subseteq \\{|k|\\le k_{\\max}\\}$. Then
$v_{\\mathrm{geo}} = S v \\cdot m$ satisfies
$\\mathrm{supp}\\,\\mathcal{F}[v_{\\mathrm{geo}}] \\not\\subseteq
\\{|k|\\le k_{\\max}\\}$: for every $\\kappa$ with
$\\mathcal{F}[Sv](\\kappa) \\neq 0$ and every $k$ with $|k| > k_{\\max}$ and
$\\mathcal{F}[m](k - \\kappa) \\neq 0$,
$\\mathcal{F}[v_{\\mathrm{geo}}](k) \\supseteq
\\mathcal{F}[Sv](\\kappa)\\,\\mathcal{F}[m](k-\\kappa)$ as a summand of the
convolution. The restoration costs $O(N \\log N)$ (two extra FFTs of width
$c$), and the new high-band content is *geometry-localized* since $m$ is a
function of the SDF.

*Proof.* Multiplication in physical space is convolution in frequency:
$\\mathcal{F}[Sv \\cdot m] = \\mathcal{F}[Sv] * \\mathcal{F}[m]$, i.e.
$\\mathcal{F}[v_{\\mathrm{geo}}](k) = \\sum_\\kappa
\\mathcal{F}[Sv](\\kappa)\\,\\mathcal{F}[m](k-\\kappa)$. The summand displayed
is nonzero for the stated $(\\kappa, k)$, so the coefficient at $k$ is
nonzero unless cancelled by other summands — which requires the same
measure-zero coincidence as in Lemma 3. Cost: $m$ is computed with one FFT
+ one inverse FFT (`sdf_conv`) plus a 1$\\times$1 conv; the multiplication is
$O(N^2)$. $\\square$

**Corollary (necessity of field-valued modulation).** If $m \\equiv$ const
(in particular at gate zero, $g_s = 0$), then
$\\mathcal{F}[v_{\\mathrm{geo}}]$ has exactly zero energy above the cut:
band restoration is *caused* by the modulation being a field, not a scalar.
This is unit-tested (`test_agf_modulation_restores_high_band_content`:
$>20\\times$ high-band gain with a learned field, $0$ with a constant).

**Remark (why modulation, not injection).** Additive geometry injection
($v + W_m g$) and FiLM affine modulation change the *latent* but leave every
block's global channel exactly as in Theorem 1: band-limited, sinc-shaped,
geometry-blind. Injected geometry content added to the latent is then
annihilated by the next block's truncation in the global path (Lemma 1) and
must re-enter through pointwise paths. Multiplicative field modulation is
the minimal change that alters the global channel itself: it turns the
shift-invariant kernel into a geometry-adaptive one ($S$ composed with
pointwise multiplication by an SDF-derived field) and provably re-populates
the truncated band (Theorem 2) before the next block's truncation. This is
the precise structural delta between AGF-NO and injection-style remedies.

**Remark (the probe is a lower bound).** The linear probe's $R^2$ lower-bounds
the decodable geometry: any nonlinear decoder can only do better. The
forgetting *rates* we report (collapse to $R^2 \\approx 0$ at depth 16, AGF
plateau at $\\approx 0.63$) are therefore conservative estimates of the
underlying information dynamics; the DPI argument of Xia & Aviles-Rivero
(2026) gives the matching upper-bound phenomenon (information cannot
increase without re-injection), and our re-injection is exactly the case
their Proposition 3.1 excludes.
