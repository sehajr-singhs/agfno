# AGF-NO — Restoring the Truncated Band

**Geometry-aware Fourier Neural Operators, and a direct measurement of geometric forgetting in 2-D and 3-D.**

Standard Fourier Neural Operators learn in the frequency domain with O(N log N) cost, but their only global mixing channel is *structurally band-limited*: mode truncation discards every Fourier mode above the cut, so boundary content must either be re-synthesized by pointwise nonlinearities or be lost. **AGF-NO** fixes the mechanism rather than the symptom: a multiplicative, field-valued modulation of the spectral weights — derived from a signed distance function (SDF) of the domain — restores the missing band through spectral convolution itself. The layer is zero-gated, so the network begins as an exact FNO.

Then the paper does what the operator-learning literature had not done for "geometric forgetting": **measures it** — with a closed-form linear probe on every block's latent features, to the point of architectural failure, in 2-D and 3-D.

## Headline results

**2-D obstacle Darcy** (48², 3 seeds, identical budgets):

| rel-L² | depth 4 | depth 8 | depth 16 |
|---|---|---|---|
| FNO | 0.125 ± 0.003 | 0.145 ± 0.004 | **1.005 ± 0.002 — collapsed** |
| AGF-NO | **0.080 ± 0.001** | **0.111 ± 0.006** | **0.084 ± 0.001 — depth-stable** |

Near-wall (ring) error: **0.251 → 0.112** at depth 4. Linear-probe R² for SDF decodability at depth 16: FNO **0.0003** (geometry erased) vs AGF-NO **0.633** (plateau). Same story on the canonical piecewise-Darcy benchmark (−25% global) with every confound removed.

**3-D obstacle Darcy** (32³, spheres *and* solid tori — multiply-connected, so deformation-style escapes are structurally unavailable; 3 seeds):

| rel-L² | depth 4 | depth 16 |
|---|---|---|
| FNO | 0.144 ± 0.020 | **1.008 ± 0.000 — collapsed, zero seed overlap** |
| AGF-NO | **0.077 ± 0.003** | **0.113 ± 0.003 — depth-stable** |

Probe R² at depth 16: FNO **0.020** vs AGF-NO **0.494**. The collapse–plateau phenomenon is dimension-independent.

## The evidence stack

- **Theory.** The global channel of an FNO block *is* the low-band projection; the band it cannot carry is exactly the band geometry needs. Multiplicative field modulation provably re-populates the truncated band (the restoration identity); additive/FiLM-style injection does not.
- **Measurement.** A closed-form ridge probe quantifies decodable geometry per block. FNO: 0.90 → 0.47 over four blocks, → 0.0003 at 16. AGF-NO: 0.93 → 0.59, → 0.633 plateau.
- **Controls.** Zero-gate identity at init (the exact-FNO equivalence, tested), capacity-matched frozen variant, spec-only ablation, penalty removed (both directions), identical inputs and budgets everywhere.
- **Cross-PDE.** Advection family: the error is boundary-resolved there too; zero-shot rollout ratio ~1.44 for both — the fix adds fidelity, not fragility.
- **Diagnostic.** A closed-form, a-priori Δρ predicts where the mechanism pays (validated on both PDE families, on the same data the models see).
- **Baselines.** Geo-FNO-style learned deformation: improves global error (not simply connected → structurally handicapped), blind at the walls. Published FNO/Geo-FNO numbers quoted for scale only, with explicit non-comparability caveats.
- **3-D capstone.** The full 12-cell matrix (2 models × 2 depths × 3 seeds) on spheres + tori.
- **Frequency-resolved attribution.** High-band ring error −43%; sinc-shaped propagation footprints (P1); >20× high-band energy at walls for the modulated vs constant kernel.

Every number in the paper is a macro typeset programmatically from `results/*.json` (`paper/macros.tex`, 221 macros) — the same JSONs committed here.

## Reproduce

```bash
git clone https://github.com/sehajr-singhs/agfno
cd agfno
pip install -r requirements.txt

# 41 unit tests: solvers, SDFs, the zero-gate identity, gates, probe
python -m pytest agfno/tests -q

# CPU smoke: data → 1 epoch → eval → probe, every experiment family
python smoke_suite.py

# 2-D controlled matrix + forgetting sweep (GPU: Modal A100 or local CUDA)
python launch.py            # full pipeline
python launch.py --quick    # sanity run

# 3-D matrix (FNO/AGF-NO × depth 4/16 × 3 seeds, T4-scale)
python -m agfno.experiments4 --quick
```

Trained models, checkpoints, and the full artifact stack: [huggingface.co/Sejibeji/agfno-darcy](https://huggingface.co/Sejibeji/agfno-darcy). Result JSONs also mirrored on Kaggle.

## Repository map

```
agfno/            package
  models.py       FNO2d / AGFNO2d (zero-gated SDF modulation + injection)
  models3d.py     3-D counterparts
  dataset.py      2-D obstacle Darcy: GRF fields, polygon SDFs, multigrid-PCG solver
  dataset3d.py    3-D: spheres + tori, float64 Jacobi-PCG solver
  probe.py        closed-form ridge probe (decodable geometry per block)
  analysis.py     frequency-resolved attribution, sinc footprints
  diagnostic.py   a-priori Δρ diagnostic
  experiments*.py controlled matrix, PDE2, depth sweep (3 seeds), 3-D matrix,
                  matched external baselines (U-Net + CNO, parameter-matched) and the
                  pilot-budget four-way suite (experiments5p)
  baselines.py    parameter-matched U-Net + faithful CNO (width auto-bisected to budget)
  tests/          43 tests
paper/            main.tex (16 pp), macros.tex (250+ programmatic macros), compiled PDF,
                  gen_hero.py (regenerates the qualitative hero figure from the HF checkpoints)
results/          committed run JSONs (12 cells × 3 seeds) + summaries
figs/             paper figures (PNG), incl. hero_qualitative.png (depth-16 FNO 1.010 vs
                  AGF-NO 0.060 on the median held-out case; boundaries outlined in cyan)
```

## License

MIT — see [LICENSE](LICENSE).
