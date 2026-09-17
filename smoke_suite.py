"""Tiny CPU smoke test for the controlled-experiment + benchmark suites.

The real suites are GPU workloads (16+ trainings). Here we shrink *everything*
-- model width, modes, dataset size, epochs, seed count, depth sweep -- so the
whole orchestration (data -> train -> eval -> aggregate -> figures -> json)
executes in a couple of CPU minutes and reveals wiring bugs before we burn a
Kaggle GPU.

Usage:
    python smoke_suite.py exp        # experiments suite only
    python smoke_suite.py bench      # canonical benchmark only
    python smoke_suite.py all        # both (default)
"""

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agfno.config as C

# ---- Shrink the model so CPU training is fast -----------------------------
C.MODEL.width = 8
C.MODEL.modes_h = 4
C.MODEL.modes_w = 4

# ---- Shrink the controlled-experiment matrix ------------------------------
C.EXPERIMENT.res = 16
C.EXPERIMENT.n_train = 64
C.EXPERIMENT.n_val = 32
C.EXPERIMENT.n_test = 32
C.EXPERIMENT.n_test_hi = 0
C.EXPERIMENT.epochs = 1
C.EXPERIMENT.depth_epochs = 1
C.EXPERIMENT.seeds = (0,)
C.EXPERIMENT.depths = (1, 2)

# ---- Shrink the canonical benchmark ---------------------------------------
C.BENCHMARK.res = 12
C.BENCHMARK.n_train = 48
C.BENCHMARK.n_val = 16
C.BENCHMARK.n_test = 24
C.BENCHMARK.epochs = 1

# ---- Shrink PDE2 (advection-diffusion) ------------------------------------
C.ADVEC.res = 16
C.ADVEC.n_train = 64
C.ADVEC.n_val = 32
C.ADVEC.n_test = 32

C.TRAIN.batch_size = 16

# ---- Shrink experiments3 (multi-seed depth sweep) -------------------------
C.EXPERIMENT.depth_epochs = 1

# ---- Shrink experiments4 (3-D replication) --------------------------------
C.DATA3D.n_train = 24
C.DATA3D.n_val = 12
C.DATA3D.n_test = 12

ROOT = "/tmp/agfno_suite"
shutil.rmtree(ROOT, ignore_errors=True)


def _show(path, label):
    """Print the aggregated summary so we can eyeball the wiring end to end."""
    full = os.path.join(path, "summary.json")
    with open(full) as f:
        s = json.load(f)
    print(f"\n=== {label} -> {full} ===")
    print(json.dumps(s, indent=2)[:2200])


which = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()

if which in ("exp4", "all3"):
    from agfno import experiments4  # noqa: E402

    t0 = time.time()
    out5 = experiments4.run_all(ROOT, quick=True)
    print(f"\nEXPERIMENTS4 (3-D) OK in {(time.time()-t0)/60:.2f} min -> {out5['run_dir']}")
    _show(out5["run_dir"], "experiments4")

if which in ("exp3", "all3"):
    from agfno import experiments3  # noqa: E402

    t0 = time.time()
    out4 = experiments3.run_all(ROOT, quick=True)
    print(f"\nEXPERIMENTS3 OK in {(time.time()-t0)/60:.2f} min -> {out4['run_dir']}")
    _show(out4["run_dir"], "experiments3")

if which in ("exp2", "all2"):
    from agfno import experiments2  # noqa: E402

    t0 = time.time()
    out3 = experiments2.run_all(ROOT, quick=True)
    print(f"\nEXPERIMENTS2 OK in {(time.time()-t0)/60:.2f} min -> {out3['run_dir']}")
    _show(out3["run_dir"], "experiments2")

if which in ("exp", "all", "all2"):
    from agfno import experiments  # noqa: E402

    t0 = time.time()
    out1 = experiments.run_all(ROOT, quick=True)
    print(f"\nEXPERIMENTS OK in {(time.time()-t0)/60:.2f} min -> {out1['run_dir']}")
    _show(out1["run_dir"], "experiments")

if which in ("bench", "all", "all2"):
    from agfno import benchmark  # noqa: E402

    t0 = time.time()
    out2 = benchmark.main(ROOT, quick=True)
    print(f"\nBENCHMARK OK in {(time.time()-t0)/60:.2f} min -> {out2['run_dir']}")
    _show(out2["run_dir"], "benchmark")

print("\nSMOKE SUITE COMPLETE")
