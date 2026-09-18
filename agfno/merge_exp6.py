"""Merge the two split exp6 kernel outputs into one summary + gates.

The exp6 sweep runs as two concurrent Kaggle kernels (one per arm: afno,
agfafno) so the 18-cell matrix fits the session budget. After pulling both
outputs, this script rebuilds the union aggregate, recomputes the cross-arm
gates (A3 rescue ratio, A4 probe mechanism), and writes:

    <out>/runs/experiments6/summary.json   (merged: agg + gates + probe)
    <out>/runs/experiments6/afno_forgetting_curve.png
    <out>/exp6_gates.json                  (flat, for the paper generator)

Usage:
    python -m agfno.merge_exp6 <dir_arm_afno> <dir_arm_agfafno> --out <out>

Each arm dir is a kernel-output folder containing runs/experiments6/ with
per-cell run_results.json files and its own summary.json (per-arm probe).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

METRIC_KEYS = ("rel_l2", "rel_l2_ring", "wall_viol")


def _load_cells(exp_dir: str) -> dict:
    """(depth, label) -> {seed: metrics} from per-cell run_results.json."""
    runs: dict = {}
    for path in sorted(glob.glob(os.path.join(exp_dir, "d*_s*/run_results.json"))):
        rec = json.load(open(path))
        runs.setdefault((int(rec["depth"]), rec["model"]), {})[int(rec["seed"])] = {
            k: rec[k] for k in METRIC_KEYS
        }
    return runs


def _aggregate(runs: dict) -> dict:
    agg = {}
    for (depth, label), per_seed in sorted(runs.items()):
        mets = list(per_seed.values())
        cell = {"n_seeds": len(mets), "depth": depth, "model": label}
        for k in METRIC_KEYS:
            vals = [m[k] for m in mets]
            cell[f"{k}_mean"] = float(np.nanmean(vals))
            cell[f"{k}_std"] = float(np.nanstd(vals))
            if len(mets) > 1:
                cell[f"{k}_per_seed"] = [float(v) for v in vals]
        agg[f"d{depth}_{label}"] = cell
    return agg


def merge(dir_afno: str, dir_agfafno: str, out: str) -> dict:
    exp_dirs = [
        os.path.join(dir_afno, "runs", "experiments6"),
        os.path.join(dir_agfafno, "runs", "experiments6"),
    ]
    runs: dict = {}
    for d in exp_dirs:
        runs.update(_load_cells(d))
    if not runs:
        raise SystemExit(f"no run_results.json found under {exp_dirs}")
    agg = _aggregate(runs)

    depths = sorted({int(k.split("_")[0][1:]) for k in agg})
    gates: dict = {}
    if "d16_afno" in agg:
        f16 = agg["d16_afno"]
        gates["A1_afno16_mean_rel_l2"] = f16["rel_l2_mean"]
        gates["A1_afno16_collapsed"] = bool(f16["rel_l2_mean"] >= 0.9)
        gates["A1_afno16_worst_seed_best_case"] = float(min(f16["rel_l2_per_seed"]))
    if "d16_agfafno" in agg:
        a16 = agg["d16_agfafno"]
        gates["A2_agfafno16_is_best_agf_cell"] = bool(
            a16["rel_l2_mean"]
            <= min(agg[f"d{d}_agfafno"]["rel_l2_mean"] for d in depths) + 1e-6
        )
        gates["A2_agfafno16_mean_rel_l2"] = a16["rel_l2_mean"]
    if "d16_afno" in agg and "d16_agfafno" in agg:
        f16, a16 = agg["d16_afno"], agg["d16_agfafno"]
        gates["A3_transplant_ratio"] = float(f16["rel_l2_mean"] / a16["rel_l2_mean"])
        gates["A3_rescued"] = bool(gates["A3_transplant_ratio"] >= 2.0)
        gates["A3_afno16_vs_agfafno16_ring_ratio"] = float(
            f16["rel_l2_ring_mean"] / a16["rel_l2_ring_mean"])

    # Probe union from the per-arm summaries
    probe = {"final_block_r2_by_depth": {}, "r2_drop_depth4_to_16": {}}
    for d in exp_dirs:
        sp = os.path.join(d, "summary.json")
        if not os.path.exists(sp):
            continue
        arm = json.load(open(sp)).get("probe", {})
        probe["final_block_r2_by_depth"].update(arm.get("final_block_r2_by_depth", {}))
    for m, pts in probe["final_block_r2_by_depth"].items():
        ds = sorted(int(k) for k in pts)
        if len(ds) >= 2:
            probe["r2_drop_depth4_to_16"][m] = float(pts[str(ds[0])] - pts[str(ds[-1])])
    drops = probe["r2_drop_depth4_to_16"]
    if {"afno", "agfafno"} <= set(drops):
        gates["A4_probe_afno_drop"] = drops["afno"]
        gates["A4_probe_agfafno_drop"] = drops["agfafno"]
        gates["A4_mechanism_shown"] = bool(drops["afno"] > drops["agfafno"])

    core = [gates[k] for k in ("A1_afno16_collapsed", "A2_agfafno16_is_best_agf_cell",
                               "A3_rescued", "A4_mechanism_shown") if k in gates]
    if core:
        gates["ALL_GATES_PASS"] = bool(all(core))

    merged = {"depth": agg, "gates": gates, "probe": probe,
              "config": {"depths": depths, "seeds": [0, 1, 2],
                         "source": "merged from split kernels"}}
    exp_out = os.path.join(out, "runs", "experiments6")
    os.makedirs(exp_out, exist_ok=True)
    with open(os.path.join(exp_out, "summary.json"), "w") as f:
        json.dump(merged, f, indent=2)
    with open(os.path.join(out, "exp6_gates.json"), "w") as f:
        json.dump(gates, f, indent=2)

    # Figure via the suite's plotter (import avoids duplicating the style).
    from .experiments6 import _figure

    _figure(agg, os.path.join(exp_out, "afno_forgetting_curve.png"))

    print("[merge_exp6] cells:", " ".join(sorted(agg)))
    print("[merge_exp6] gates:", json.dumps(gates, indent=1))
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_afno")
    ap.add_argument("dir_agfafno")
    ap.add_argument("--out", default="kaggle/exp6_out")
    a = ap.parse_args()
    merge(a.dir_afno, a.dir_agfafno, a.out)
