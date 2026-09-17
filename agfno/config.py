"""Central configuration for the AGF-NO research project.

Every hyper-parameter used anywhere in the pipeline lives here so that any
run can be reproduced exactly by pinning this file (or overriding the CLI
flags exposed by ``train.py``).

Scientific scope
----------------
We compare a standard Fourier Neural Operator (FNO, Li et al. 2021) against
our Adaptive Geometry-Aware Fourier Neural Operator (AGF-NO) on a 2-D Darcy
flow benchmark with *random heterogeneous permeability fields* and *random
irregular polygonal obstacles* carved out of the periodic box with exact
Dirichlet (no-slip) boundary conditions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 0


# --------------------------------------------------------------------------- #
# Data generation: 2-D Darcy flow with obstacles
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    # Spatial resolution of the uniform grid (res_h x res_w, square here).
    res: int = 48
    # Number of samples per split.
    n_train: int = 2000
    n_val: int = 200
    n_test: int = 1000
    # Canonical piecewise-constant Darcy benchmark (Li et al. 2021 family):
    # the domain is split into n_squares x n_squares equal cells, each with a
    # random circle; K = k_high inside a circle, k_low outside. Flow passes
    # through the circles (no wall conditions) -- the standard benchmark.
    pw_n_squares: int = 4
    pw_k_high: float = 12.0
    pw_k_low: float = 1.0
    pw_radius_range: tuple = (0.06, 0.14)
    # Obstacles: number of polygonal "rocks" per sample (irregular geometry).
    n_obstacles_range: tuple = (1, 3)
    # Obstacle size range, expressed as a fraction of the domain side length.
    obstacle_radius_range: tuple = (0.10, 0.22)
    # Log-normal permeability field: log K ~ smooth Gaussian random field with
    # unit variance; K = exp(lognormal_scale * a(x, y)) is heterogeneous.
    lognormal_scale: float = 0.6
    # Number of Fourier modes used to synthesize the random coefficient /
    # obstacle geometry. Larger => rougher fields.
    n_random_fourier_modes: int = 12
    # Dirichlet value imposed on obstacle boundaries (no-flow / no-slip wall).
    obstacle_dirichlet_value: float = 0.0
    # Normalization of inputs/outputs computed from the training split.
    eps: float = 1e-6

    # File names (cached on the Modal volume so data is generated once).
    def shard_name(self, split: str, n: int, res: int, kind: str = "obstacles") -> str:
        return f"darcy_{kind}_{split}_n{n}_res{res}_s{SEED}.npz"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    # Input channels: [log K, signed distance to boundary, boundary ring mask].
    in_ch: int = 3
    out_ch: int = 1
    # Hidden width and depth (number of Fourier blocks). The paper's ablation
    # grid is width x depth x modes; we use the strongest setting that fits a
    # single A100 comfortably at res 96 evaluation.
    width: int = 64
    n_blocks: int = 4
    # Fourier modes retained per axis (truncation, the "mode cut" of the FNO).
    modes_h: int = 12
    modes_w: int = 12
    # MLP hidden factor inside each block (v . W(L(...)) branch).
    mlp_ratio: int = 2
    dropout: float = 0.0
    # Geometry injection: signed-distance channels re-injected in every block.
    sdf_ch: int = 3
    # SDF-gated spectral conv: the learnable strength is a scalar per layer,
    # initialized at 0 so AGF-NO starts as an exact FNO and learns to use
    # geometry. The gate is clamped to this max gain for numerical stability.
    gate_max: float = 5.0
    # Channels added to the latent state for the geometry stream:
    # [sdf, ring mask, interior mask] plus Fourier-feature encoding of coords.
    use_coord_features: bool = True
    n_coord_features: int = 4  # sin/cos of 2pi x and 2pi y


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    batch_size: int = 64
    num_epochs: int = 300
    # Relative L2 (energy) loss is the standard metric for operator learning;
    # boundary_lmbda weights the extra boundary-ring penalty of AGF-NO.
    boundary_lmbda: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # Cosine annealing of the learning rate over the full schedule.
    warmup_frac: float = 0.03
    # Evaluate every N epochs on the validation split; keep the best ckpt.
    eval_every: int = 10
    # Number of test samples visualized in the comparison figure.
    n_vis: int = 3


# --------------------------------------------------------------------------- #
# Infrastructure: Modal / Hugging Face / Kaggle
# --------------------------------------------------------------------------- #
@dataclass
class InfraConfig:
    app_name: str = "agfno-darcy"
    volume_name: str = "agfno-vol"
    volume_mount: str = "/vol"
    gpu: str = "A100"
    timeout_s: int = 4 * 3600
    cpu_cores: float = 8.0
    # Hugging Face repository that receives checkpoints / metrics / figures.
    hf_repo: str = field(
        default_factory=lambda: os.environ.get("AGFNO_HF_REPO", "sehaj fats")
        .strip()
        .replace(" ", "-")
    )


# --------------------------------------------------------------------------- #
# Controlled-experiment suite (ablation / multi-seed / depth sweep)
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    # Shared dataset for every run (test split is byte-identical to the full
    # research run, so metrics are directly comparable across experiments).
    res: int = 48
    n_train: int = 1024
    n_val: int = 128
    n_test: int = 1000
    n_test_hi: int = 512
    # Training budget per run. Kept smaller than the headline run (which used
    # 300 epochs / 2000 samples) so the full matrix fits a single GPU kernel;
    # every variant shares the identical budget, keeping comparisons fair.
    epochs: int = 150
    depth_epochs: int = 120
    # Seeds for the mean +/- std statistics on the headline pair.
    seeds: tuple = (0, 1, 2)
    # Depth sweep for the geometric-forgetting curve (depth 4 = the main runs).
    depths: tuple = (1, 2, 3)


# --------------------------------------------------------------------------- #
# Canonical benchmark (piecewise-constant Darcy, compared with the literature)
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkConfig:
    # Grid resolution. The literature reports 85x85; we use 64x64 (cheaper on
    # one T4) and note the difference in the results file.
    res: int = 64
    n_train: int = 2000
    n_val: int = 256
    n_test: int = 1000
    epochs: int = 200
    # Published reference numbers on the standard Darcy benchmark (85x85,
    # 10k training samples), as reported in the literature -- cited in the
    # results table for context, not produced by our pipeline.
    published_fno_rel_l2: float = 0.0082  # Li et al., ICLR 2021
    published_geofno_rel_l2: float = 0.0068  # Li et al., JMLR 2023 (Geo-FNO)


# --------------------------------------------------------------------------- #
# PDE family 2: advection-diffusion past fixed-temperature obstacles
# --------------------------------------------------------------------------- #
@dataclass
class AdvectionConfig:
    """Second PDE family: scalar advection-diffusion through the SAME class of
    irregular polygonal obstacle fields as Darcy, but with *dynamics*:

        du/dt + v(x) . grad u  =  nu * Lap u      in the fluid
        u = 0 (ambient)                            on obstacle interiors

    The velocity is divergence-free by construction (v = curl(psi) for a
    smooth random stream function), held steady in time, so the learned
    operator  (u_0, v, geometry) -> u_T  is autonomous and *composable*:
    applying it twice must extrapolate to 2T (zero-shot rollout in time --
    the time-analogue of the spatial super-resolution axis).

    Why this family is hard for a vanilla FNO: advection of hot fluid past
    cold walls builds thermal boundary layers of thickness ~ sqrt(nu L / |v|)
    (~4-6 cells at res 48) that are SHARPER than anything in the elliptic
    problem and whose position/shape is dictated by the geometry -- exactly
    the high-band, geometry-coupled content that mode truncation destroys.

    Numerics: backward semi-Lagrangian advection (unconditionally stable,
    monotone via bilinear sampling) + implicit diffusion solved with the same
    multigrid-preconditioned PCG used for Darcy (the diffusion system is
    SPD: I + dt*nu*L + penalty*M_interior). Obstacles act as fixed-temperature
    (ambient) inclusions -- an idealization we state openly in the paper.
    """
    res: int = 48
    n_train: int = 1024
    n_val: int = 128
    n_test: int = 512
    # Time integration.
    dt: float = 0.05
    nu: float = 0.005          # diffusivity (Pe = |v| L / nu ~ 100)
    n_steps_train: int = 10    # operator horizon T = 10 dt = 0.5
    n_steps_long: int = 20     # zero-shot rollout horizon 2T
    # Velocity: v = curl(psi), psi a smooth random stream function.
    psi_modes: int = 16
    vel_rms: float = 0.5       # RMS speed target
    # Initial condition: mixture of positive Gaussian hot spots.
    n_bumps_range: tuple = (3, 8)
    bump_amp_range: tuple = (0.5, 1.5)
    bump_sigma_range: tuple = (0.05, 0.15)

    def shard_name(self, split: str, n: int, res: int) -> str:
        return f"advec_{split}_n{n}_res{res}_s{SEED}.npz"


# --------------------------------------------------------------------------- #
# 3-D Darcy flow through solid obstacles (spheres + solid tori)
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig3D:
    """The 2-D obstacle benchmark lifted to 3-D: random heterogeneous
    permeability + spherical/solid-torus obstacles with exact Dirichlet
    walls. Tori keep the domain multiply-connected in 3-D (no diffeomorphism
    of the box flattens an interior toroidal wall).

    Sizes are set for a single T4 (16 GB): res 32 training with zero-shot
    super-resolution to 48; solver cost scales as N log N per CG iteration
    with N = res^3 (32^3 = 32,768 unknowns).
    """
    res: int = 32
    res_hi: int = 48
    n_train: int = 512
    n_val: int = 64
    n_test: int = 256
    # Obstacles per sample: spheres and (occasionally) one solid torus.
    n_obstacles_range: tuple = (1, 3)
    obstacle_radius_range: tuple = (0.10, 0.20)
    torus_major_range: tuple = (0.18, 0.26)
    torus_minor_range: tuple = (0.06, 0.10)
    p_torus: float = 0.35  # probability that a sample contains a torus
    lognormal_scale: float = 0.6
    n_random_fourier_modes: int = 8
    obstacle_dirichlet_value: float = 0.0
    eps: float = 1e-6

    def shard_name(self, split: str, n: int, res: int) -> str:
        return f"darcy3d_{split}_n{n}_res{res}_s{SEED}.npz"


@dataclass
class ModelConfig3D:
    """3-D twin of ModelConfig. Width is reduced (32 vs 64) and the mode cut
    per axis is reduced (8 vs 12) so a depth-16 model still fits a T4 with
    batch 8 at res 32: a SpectralConv3d weight tensor is
    in_ch*out_ch*modes_d*modes_h*modes_w complex numbers, and 3-D FFTs are
    the dominant memory/time cost."""
    in_ch: int = 3
    out_ch: int = 1
    width: int = 32
    n_blocks: int = 4
    modes_d: int = 8
    modes_h: int = 8
    modes_w: int = 8
    mlp_ratio: int = 2
    dropout: float = 0.0
    sdf_ch: int = 3
    gate_max: float = 5.0
    use_coord_features: bool = True
    n_coord_features: int = 6  # sin/cos of 2pi x, y, z


# Convenience singletons -----------------------------------------------------
DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
INFRA = InfraConfig()
EXPERIMENT = ExperimentConfig()
BENCHMARK = BenchmarkConfig()
ADVEC = AdvectionConfig()
DATA3D = DataConfig3D()
MODEL3D = ModelConfig3D()
