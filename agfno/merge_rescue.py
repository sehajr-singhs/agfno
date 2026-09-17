"""Merge rescue-kernel outputs into the final 3-D matrix summary.

The original exp4 kernel completed the six depth-4 cells before dying of
host OOM; the two rescue kernels (exp4r-fno / exp4r-agfno) each trained
one model's three depth-16 runs. Neither kernel sees the full matrix, so
the stability gates are computed here, locally, from all 12 cells.

Usage (after pulling both kernels' output dirs into kaggle/exp4_out_r):

    python -m agfno.merge_rescue --fno kaggle/exp4_out_r/fno \
                                 --agfno kaggle/exp4_out_r/agfno \
                                 --out kaggle/exp4_out
"""

from __future__ import annotations

import argparse
import json
import os

from agfno.experiments4 import _METRIC_KEYS, _aggregate_and_gates, _figure


def _load_runs(root: str) -> tuple[dict, dict | None]:
    """Load per-cell metric dicts from an exp4-style runs directory.

    Returns ({(depth, model): {seed: metrics}}, config-or-None) from the
    saved run_results.json files, mirroring the in-kernel representation
    before aggregation. The kernel's summary.json (if present) supplies the
    ``config`` block for the paper macros.
    """
    runs: dict = {}
    exp_root = os.path.join(root, "runs", "experiments4")
    if not os.path.isdir(exp_root):
        raise SystemExit(f"missing runs dir: {exp_root}")
    for d in sorted(os.listdir(exp_root)):
        path = os.path.join(exp_root, d, "run_results.json")
        if not os.path.isfile(path):
            continue
        saved = json.load(open(path))
        depth, label, seed = saved["depth"], saved["model"], saved["seed"]
        m = {k: saved[k] for k in _METRIC_KEYS
             if k in saved and isinstance(saved[k], (int, float))}
        runs.setdefault((depth, label), {})[seed] = m
    cfg = None
    sum_path = os.path.join(exp_root, "summary.json")
    if os.path.isfile(sum_path):
        cfg = json.load(open(sum_path)).get("config")
    return runs, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fno", required=True,
                    help="exp4r-fno kernel output dir")
    ap.add_argument("--agfno", required=True,
                    help="exp4r-agfno kernel output dir")
    ap.add_argument("--out", required=True,
                    help="output dir for the merged summary + figure")
    args = ap.parse_args()

    runs = {}
    configs = []
    for root in (args.fno, args.agfno):
        cell_runs, cfg = _load_runs(root)
        for cell, seeds in cell_runs.items():
            runs.setdefault(cell, {}).update(seeds)
        if cfg:
            configs.append(cfg)

    n_cells = len(runs)
    agg, gates = _aggregate_and_gates(runs, quick=False)
    # config: prefer the richest kernel summary; fall back to a minimal block
    cfg = max(configs, key=len, default=None) or {
        "seeds": sorted({s for (_, _), seeds in runs.items()
                         for s in seeds}),
    }
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"depth": agg, "gates": gates, "config": cfg,
                   "merged_from": [args.fno, args.agfno]}, f, indent=2)
    _figure(agg, os.path.join(args.out, "forgetting3d.png"))

    print(f"[merge_rescue] merged {n_cells} cells from:")
    for (depth, label), seeds in sorted(runs.items()):
        for seed in sorted(seeds):
            print(f"  d{depth}_{label}_s{seed}")
    print("[merge_rescue] gates:", json.dumps(gates, indent=2))
    print(f"[merge_rescue] wrote {out}")


if __name__ == "__main__":
    main()
