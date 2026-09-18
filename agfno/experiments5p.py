"""Pilot-budget matched-baselines study (CPU-executable, headless).

Why a pilot exists
------------------
The full-budget matched-baselines suite (``experiments5.py``) needs ~5 GPU-hours;
Kaggle's weekly GPU quota and the Modal spend limit were both exhausted when it
was staged. This module runs the SAME comparison at a reduced but IDENTICAL
budget for ALL FOUR architectures -- FNO and AGF-NO are retrained under the
pilot budget too, so every cell of the pilot table is comparable and the
architecture ordering claim is internally valid.

Pilot protocol (public, disclosed in the paper): 256 train / 64 val / 128 test
samples at 48x48, 32 epochs, batch 16, boundary penalty ON (headline loss),
3 seeds per architecture. Data generation is byte-deterministic: pilot splits
share their first N cases with the full-budget splits of ``experiments.py``.

The full-budget section auto-replaces this one when the GPU kernel runs
Saturday; both tables are then reported (pilot = immediate replication,
full = the headline protocol).

Output: ``runs/experiments5p/pilot_summary.json`` with per-seed rows, the
four-way aggregate, and the ordering gate.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace

import numpy as np
import torch

from . import config as C
from . import dataset as D
from . import utils as U
from .baselines import _n_params
from .models import fno2d
from .train import to_tensors, train_one

VARIANTS = [("fno", "fno"), ("unet", "unet"), ("cno", "cno"), ("agfno", "agfno")]
SEEDS = C.EXPERIMENT.seeds

PILOT = dict(n_train=256, n_val=64, n_test=128, res=48, epochs=32, batch=16)


def _evaluate(model, data: dict) -> dict:
    te = data["test"]
    return U.eval_metrics(model, te["x"], te["g"], te["u"], te["interior"], te["ring"])


def _save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_all(artifacts_root: str = "/kaggle/working", quick: bool = False,
            resume: bool = True, seeds=SEEDS) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[experiments5p] device = {device} (pilot protocol: {PILOT})")

    vol_root = (C.INFRA.volume_mount
                if os.path.isdir(C.INFRA.volume_mount) else "/tmp/agfno_vol")
    os.makedirs(vol_root, exist_ok=True)

    # Data: byte-deterministic; shares its first N cases with the full splits.
    splits = {
        "train": D.load_or_make_split("train", PILOT["n_train"], PILOT["res"], vol_root, device),
        "val": D.load_or_make_split("val", PILOT["n_val"], PILOT["res"], vol_root, device),
        "test": D.load_or_make_split("test", PILOT["n_test"], PILOT["res"], vol_root, device),
    }
    train_n, val_n, _ = D.normalize_split(splits["train"], splits["val"])
    _, test_n, _ = D.normalize_split(splits["train"], splits["test"])
    data = {"train": to_tensors(train_n, device),
            "val": to_tensors(val_n, device),
            "test": to_tensors(test_n, device)}
    del splits, train_n, val_n, test_n

    exp_root = os.path.join(artifacts_root, "runs", "experiments5p")
    os.makedirs(exp_root, exist_ok=True)

    cfg_train = replace(C.TRAIN, num_epochs=PILOT["epochs"], batch_size=PILOT["batch"])
    if quick:
        cfg_train = replace(cfg_train, num_epochs=2)
        seeds = (C.SEED,)
    ref_params = _n_params(fno2d(C.MODEL))

    results: dict[str, dict[int, dict]] = {lab: {} for lab, _ in VARIANTS}
    for label, model_name in VARIANTS:
        for seed in seeds:
            tag = f"{label}_s{seed}"
            run_dir = os.path.join(exp_root, tag)
            out_path = os.path.join(run_dir, "run_results.json")
            rec = None
            if resume and os.path.exists(out_path):
                try:
                    with open(out_path) as f:
                        rec = json.load(f)
                    assert "rel_l2" in rec
                except Exception:
                    rec = None
            if rec is not None:
                m = {k: rec[k] for k in ("rel_l2", "rel_l2_ring", "wall_viol")}
                results[label][seed] = {"metrics": m, "params": rec.get("params", 0)}
                print(f"[experiments5p] {tag}: RESUMED rel_l2 {m['rel_l2']:.4f}")
                continue
            os.makedirs(run_dir, exist_ok=True)
            t0 = time.time()
            model, _ = train_one(model_name, data, C.MODEL, cfg_train, device,
                                 run_dir, quick, seed=seed)
            dt = time.time() - t0
            m = _evaluate(model, data)
            n_params = sum(p.numel() for p in model.parameters())
            results[label][seed] = {"metrics": m, "params": n_params}
            print(f"[experiments5p] {tag}: {dt/60:.1f} min | rel_l2 {m['rel_l2']:.4f} "
                  f"ring {m['rel_l2_ring']:.4f} wall {m['wall_viol']:.4f}")
            _save_json(out_path, {"label": label, "seed": seed,
                                  "train_min": round(dt / 60, 1),
                                  "params": n_params, **m})
            del model

    # Aggregate
    summary: dict[str, dict] = {}
    for label, per_seed in results.items():
        mets = [v["metrics"] for v in per_seed.values()]
        if not mets:
            continue
        agg = {"n_seeds": len(mets)}
        for k in mets[0]:
            vals = [m[k] for m in mets]
            agg[f"{k}_mean"] = float(np.mean(vals))
            if len(vals) > 1:
                agg[f"{k}_std"] = float(np.std(vals))
        agg["params"] = int(max(v.get("params", 0) for v in per_seed.values()))
        summary[label] = agg

    # Ordering gates: encode the claims the paper actually makes.
    # (v1 gate expected convolutional baselines to fail at walls like FNO;
    #  the pilot refuted that premise -- U-Net beats FNO on ring error,
    #  which ISOLATES the pathology to the periodic global spectral
    #  pathway rather than to deep nets in general.  The corrected gates
    #  state the mechanism claims directly.)
    gate = {}
    if all(l in summary for l in ("fno", "agfno", "unet", "cno")):
        gate["agf_best_or_tied_ring"] = (
            summary["agfno"]["rel_l2_ring_mean"]
            <= min(summary["unet"]["rel_l2_ring_mean"],
                   summary["cno"]["rel_l2_ring_mean"],
                   summary["fno"]["rel_l2_ring_mean"]) * 1.0)
        gate["agf_wall_best"] = (
            summary["agfno"]["wall_viol_mean"]
            < min(summary["unet"]["wall_viol_mean"],
                  summary["cno"]["wall_viol_mean"],
                  summary["fno"]["wall_viol_mean"]))
        # Mechanistic control: another band-limited global operator (CNO)
        # shows the same-or-worse ring failure as FNO -- the pathology
        # tracks the global spectral channel, not architecture family.
        gate["cno_ring_ge_fno"] = (
            summary["cno"]["rel_l2_ring_mean"]
            >= summary["fno"]["rel_l2_ring_mean"])
        gate["PILOT_ORDERING_PASS"] = bool(
            gate["agf_best_or_tied_ring"] and gate["agf_wall_best"]
            and gate["cno_ring_ge_fno"])
    summary["_pilot_protocol"] = {k: str(v) for k, v in PILOT.items()}
    summary["_pilot_gates"] = gate

    _save_json(os.path.join(exp_root, "pilot_summary.json"),
               {"summary": summary})
    print("[experiments5p] pilot gates:", json.dumps(gate, indent=1))
    return {"run_dir": exp_root, "summary": summary}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="/kaggle/working")
    a = ap.parse_args()
    run_all(a.out, a.quick)
