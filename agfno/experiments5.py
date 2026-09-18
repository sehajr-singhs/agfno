"""Matched external-baseline suite: U-Net and CNO on obstacle Darcy.

Closes Limitations (ii): no external architectures trained at matched
settings existed. This suite trains the two canonical convolutional
competitors -- U-Net (Ronneberger et al., 2015) and CNO (Raonic et al.,
2024, parameter-matched automatically to the FNO budget) -- on the
BYTE-IDENTICAL splits, budget, and eval protocol of the headline
experiments.py matrix:

* same ``D.load_or_make_split`` calls (same seed -> same bytes),
* same normalization, same ``to_tensors`` packing,
* same ``train_one`` loop (optimizer, schedule, epochs, boundary penalty,
  seeds), same ``U.eval_metrics`` scoring.

Only the architecture differs -- which is exactly the comparison a referee
asks for. Results are saved per-run as they finish, and ``resume=True``
skips any (label, seed) cell whose ``run_results.json`` already exists, so
an interrupted kernel picks up where it died instead of re-burning GPU hours.

Outputs ``runs/experiments5/summary.json`` consumed by ``paper/gen_paper.py``.
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

# (label, model_name) -- identical budget per label via train_one.
VARIANTS = [("unet", "unet"), ("cno", "cno")]
SEEDS = C.EXPERIMENT.seeds  # (0, 1, 2) in the research run


def _evaluate(model, data: dict, device: str, quick: bool) -> dict:
    te = data["test"]
    m = U.eval_metrics(model, te["x"], te["g"], te["u"], te["interior"], te["ring"])
    if not quick and "test_hi" in data:
        th = data["test_hi"]
        m = {**m, **U.super_res_eval(model, th["x"], th["g"], th["u"], device)}
    return m


def _save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_all(artifacts_root: str = "/kaggle/working", quick: bool = False,
            resume: bool = True) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[experiments5] device = {device}")
    cfg = C.EXPERIMENT

    vol_root = (
        C.INFRA.volume_mount
        if os.path.isdir(C.INFRA.volume_mount)
        else "/tmp/agfno_vol"
    )
    os.makedirs(vol_root, exist_ok=True)

    # ---------- data: byte-identical to the headline suite ----------------
    n_train = min(cfg.n_train, 256) if quick else cfg.n_train
    n_val = min(cfg.n_val, 64) if quick else cfg.n_val
    n_test = min(cfg.n_test, 64) if quick else cfg.n_test
    splits = {
        "train": D.load_or_make_split("train", n_train, cfg.res, vol_root, device),
        "val": D.load_or_make_split("val", n_val, cfg.res, vol_root, device),
        "test": D.load_or_make_split("test", n_test, cfg.res, vol_root, device),
    }
    if not quick:
        splits["test_hi"] = D.load_or_make_split(
            "test", cfg.n_test_hi, cfg.res * 2, vol_root, device
        )
    train_n, val_n, _ = D.normalize_split(splits["train"], splits["val"])
    _, test_n, _ = D.normalize_split(splits["train"], splits["test"])
    data = {
        "train": to_tensors(train_n, device),
        "val": to_tensors(val_n, device),
        "test": to_tensors(test_n, device),
    }
    if not quick:
        _, test_hi_n, _ = D.normalize_split(splits["train"], splits["test_hi"])
        data["test_hi"] = to_tensors(test_hi_n, device)
    del splits, train_n, val_n, test_n  # host-RAM hygiene (the exp4 lesson)

    exp_root = os.path.join(artifacts_root, "runs", "experiments5")
    os.makedirs(exp_root, exist_ok=True)

    epochs = 3 if quick else cfg.epochs
    cfg_train = replace(C.TRAIN, num_epochs=epochs)
    seeds = (C.SEED,) if quick else cfg.seeds

    # ---------- reference budget (logged once, for the paper) --------------
    ref_params = _n_params(fno2d(C.MODEL))
    print(f"[experiments5] reference FNO budget: {ref_params/1e6:.3f}M")

    # ---------- training ---------------------------------------------------
    results: dict[str, dict[int, dict]] = {}
    for label, model_name in VARIANTS:
        results[label] = {}
        for seed in seeds:
            tag = f"{label}_s{seed}"
            run_dir = os.path.join(exp_root, tag)
            out_path = os.path.join(run_dir, "run_results.json")
            rec = None
            if resume and os.path.exists(out_path):
                try:
                    with open(out_path) as f:
                        rec = json.load(f)
                    assert "rel_l2" in rec  # complete, not killed mid-write
                except Exception:
                    rec = None  # truncated/corrupt -> retrain this cell
            if rec is not None:
                m = {k: rec[k] for k in
                     ("rel_l2", "rel_l2_ring", "wall_viol") if k in rec}
                if "rel_l2_sr" in rec:
                    m["rel_l2_sr"] = rec["rel_l2_sr"]
                results[label][seed] = {"metrics": m,
                                        "params": rec.get("params", 0),
                                        "resumed": True}
                print(f"[experiments5] {tag}: RESUMED "
                      f"rel_l2 {m['rel_l2']:.4f}")
                continue
            os.makedirs(run_dir, exist_ok=True)
            t0 = time.time()
            model, _ = train_one(
                model_name, data, C.MODEL, cfg_train, device, run_dir,
                quick, seed=seed,
            )
            dt = time.time() - t0
            m = _evaluate(model, data, device, quick)
            n_params = sum(p.numel() for p in model.parameters())
            results[label][seed] = {"metrics": m, "params": n_params}
            print(
                f"[experiments5] {tag}: trained {dt/60:.1f} min | "
                f"rel_l2 {m['rel_l2']:.4f} ring {m['rel_l2_ring']:.4f} "
                f"wall {m['wall_viol']:.4f} "
                f"sr {m.get('rel_l2_sr', float('nan')):.4f} "
                f"params {n_params/1e6:.2f}M"
            )
            _save_json(out_path, {"label": label, "seed": seed,
                                  "train_min": round(dt / 60, 1),
                                  "params": n_params, **m})
            del model  # free GPU memory before the next cell

    # ---------- aggregation (mean/std over seeds; resume-safe) -------------
    summary: dict[str, dict] = {}
    for label, per_seed in results.items():
        mets = [v["metrics"] for v in per_seed.values() if v["metrics"]]
        if not mets:
            continue
        agg = {"n_seeds": len(mets)}
        for k in mets[0]:
            vals = [m[k] for m in mets if k in m]
            agg[f"{k}_mean"] = float(np.mean(vals))
            if len(vals) > 1:
                agg[f"{k}_std"] = float(np.std(vals))
        params = [v.get("params", 0) for v in per_seed.values()]
        agg["params"] = int(max(params))
        agg["params_vs_fno"] = round(agg["params"] / ref_params, 4)
        summary[label] = agg
    summary["_reference"] = {"fno_params": ref_params,
                             "matching_rule": "width-bisected to FNO budget",
                             "protocol": "identical to experiments.py"}

    _save_json(os.path.join(exp_root, "summary.json"),
               {"summary": summary,
                "experiment_config": {k: str(v) for k, v in cfg.__dict__.items()
                                      if not k.startswith("_")}})
    print("[experiments5] summary:")
    print(json.dumps(summary, indent=2))
    return {"run_dir": exp_root, "summary": summary}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="/kaggle/working")
    a = ap.parse_args()
    run_all(a.out, a.quick)
