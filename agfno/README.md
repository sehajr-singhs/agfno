# AGF-NO: Adaptive Geometry-Aware Fourier Neural Operator

A complete, reproducible research pipeline comparing a **standard Fourier Neural
Operator (FNO)** against a **geometry-aware variant (AGF-NO)** on 2-D Darcy flow
with random irregular obstacles — executed end-to-end on cloud GPUs
([Modal](https://modal.com), A100) with zero local compute, artifacts pushed to
the Hugging Face Hub, and the dataset mirrored to Kaggle.

## Scientific thesis

Standard FNOs assume periodic, uniform grids. We construct a benchmark where
that assumption breaks — random heterogeneous permeability fields with random
irregular polygonal obstacles under Dirichlet (no-flow) walls — and evaluate
whether injecting geometry *inside the spectral layer* fixes it:

- **AGF spectral path.** The signed-distance field of the obstacle geometry is
  pushed through its own spectral conv; the result modulates the primary
  spectral output *multiplicatively*:
  `y = (W u) * (1 + g * tanh(S(sdf)))` — a position- and frequency-dependent
  kernel that adapts to boundary geometry at O(N log N) cost.
- **Anti-forgetting injection.** Raw SDF + coordinate features re-enter every
  block's MLP branch, so boundary information never has to survive a Markovian
  chain of global mixers (the "geometric forgetting" failure mode).
- **Gated residual design.** Both geometry paths are zero-gated at init, so
  AGF-NO *starts as an exact FNO* and learns how much geometry to use
  (verified by `test_agf_block_at_zero_gate_equals_fno_block`).

## Benchmark: Darcy flow with irregular obstacles

    -div( K(x) grad u ) = 1        in the unit square minus obstacles
    u = 0                          on obstacle walls (penalty-enforced)
    outer frame sealed (K = K_eps) so the periodic FFT box is consistent

- K is a smooth log-normal random field (heterogeneous, as in Li et al. 2021).
- Obstacles: 1-3 random irregular polygons per sample (3-9 jittered vertices).
- Ground truth from a **real solver**: preconditioned conjugate gradient with a
  batched shifted-Laplacian multigrid preconditioner, residual tolerance 1e-6 —
  not a pseudo-solver. Runs fully batched on the GPU.
- Exact vectorized polygon SDFs (winding number + point-segment distance) at
  4x oversampling; identical geometry evaluated at 48 and 96 for zero-shot
  super-resolution.

## Repository layout

| File | Purpose |
|---|---|
| `agfno/config.py` | every hyper-parameter (data, model, training, infra) |
| `agfno/models.py` | `SpectralConv2d`, FNO/AGF blocks, full backbones |
| `agfno/dataset.py` | geometry, SDF, batched PCG solver, Kaggle mirror |
| `agfno/utils.py` | losses, metrics, super-res eval, figures |
| `agfno/train.py` | end-to-end orchestration (runs inside Modal) |
| `remote.py` | Modal app: image, secrets, volume, entry points |
| `launch.py` | local CLI (dispatches everything to Modal) |
| `agfno/tests/` | correctness suite (spectral, geometry, solver, losses) |

## Setup (once)

```bash
pip install modal
modal token new                                  # Modal auth
modal secret create agfno-hf HF_TOKEN=hf_xxx     # Hugging Face write token
modal secret create agfno-kaggle KAGGLE_USERNAME=... KAGGLE_KEY=...
export AGFNO_HF_REPO=<hf-username>/agfno-darcy   # where artifacts land
```

No other local installs are needed: all heavy deps live in the Modal image.

## Usage

```bash
python launch.py --envcheck   # container, GPU, secrets, HF identity
python launch.py --smoke      # CPU smoke test (data + models, seconds)
python launch.py --test       # full pytest suite on an A100
python launch.py --data-only  # generate + cache dataset, mirror to Kaggle
python launch.py --quick      # ~10 min end-to-end sanity run
python launch.py              # FULL research run (A100, ~3-5 h)
```

## What the full run produces

1. **Trained models** (FNO baseline + AGF-NO, identical budgets) and test
   metrics: relative L2, near-boundary relative L2 (ring), wall violation,
   inference throughput, zero-shot 2x super-resolution error.
2. **Figures**: `comparison.png` (GT vs FNO vs AGF-NO + error maps),
   `training_curves.png`, `super_resolution.png`.
3. **`results.json`**: configs, normalization stats, timing, full histories.
4. All of the above pushed to the Hugging Face model repo `AGFNO_HF_REPO`;
   the Darcy dataset itself mirrored to a Kaggle dataset (set `KAGGLE_SLUG`).

## Design notes

- **Complex weights, both halves.** `SpectralConv2d` learns complex matrices
  for the positive-frequency block and the negative-vertical-frequency block
  of `rfft2` output (two einsums), matching the canonical FNO.
- **Zero-shot SR without interpolation.** Fourier layers crop/zero-pad modes
  to the input grid; the pointwise paths are resolution-free, so a 48-px model
  runs on 96 px inputs directly.
- **Shifted-Laplacian PCG.** The preconditioner uses the SPD shifted system
  `(L + pI)` rather than the masked penalty operator, which keeps CG valid
  while still cutting condition number by orders of magnitude.
- **Reproducibility.** Stable per-split RNG offsets (no salted `hash()`),
  seeds pinned in `config.py`, dataset cached on a Modal volume so every run
  sees byte-identical data.
