#!/usr/bin/env python
"""Qualitative hero figure: depth-16 FNO vs AGF-NO on one held-out Darcy case.

Uses the committed exp2 checkpoints (the same ones the probe ran on), so the
figure is regenerable by anyone who clones the repo + checkpoints. The depth-16
FNO is the collapsed model (rel-L2 ~ 1.0), which is exactly the point: it
predicts an essentially geometry-free field, while AGF-NO tracks the solver.

    python paper/gen_hero.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from agfno import config as C
from agfno import dataset as D
from agfno import utils as U
from agfno.models import build_model

RUNS = os.path.join(ROOT, "kaggle", "exp2_out", "runs", "experiments2")


def load(name: str, depth: int, path: str, device: str):
    from dataclasses import replace

    cfg = replace(C.MODEL, n_blocks=depth)
    model = build_model(name, cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    U.set_seed(0)

    vol = C.INFRA.volume_mount if os.path.isdir(C.INFRA.volume_mount) else "/tmp/agfno_vol"
    te = D.load_or_make_split("test", 8, C.DATA.res, vol, device)
    _, te, _ = D.normalize_split(te, te)

    fno = load("fno", 16, os.path.join(RUNS, "d16_fno", "fno_best.pt"), device)
    agf = load("agfno", 16, os.path.join(RUNS, "d16_agfno", "agfno_best.pt"), device)

    # Honest selection rule: the held-out case whose FNO error is the MEDIAN of
    # the test batch (not the prettiest). Report every per-sample value.
    sdf = te["sdf"]

    def tensors(k: int):
        # model input x = [a, sdf, ring]; geometry stream g = [sdf, ring, interior]
        x = torch.tensor(
            np.concatenate([te["a"][k : k + 1], te["sdf"][k : k + 1],
                            te["ring"][k : k + 1]], axis=1),
            dtype=torch.float32,
        )
        g = torch.tensor(
            np.concatenate([te["sdf"][k : k + 1], te["ring"][k : k + 1],
                            te["interior"][k : k + 1]], axis=1),
            dtype=torch.float32,
        )
        return x.to(device), g.to(device)

    n = te["u"].shape[0]
    with torch.no_grad():
        rels_f = []
        preds_f = []
        preds_a = []
        for k in range(n):
            x, g = tensors(k)
            pf = fno(x, g).cpu().numpy()
            pa = agf(x, g).cpu().numpy()
            preds_f.append(pf)
            preds_a.append(pa)
            ut = te["u"][k : k + 1]
            rels_f.append(float(np.sqrt(((pf - ut) ** 2).sum())
                                / np.sqrt((ut ** 2).sum())))
    i = int(np.argsort(rels_f)[len(rels_f) // 2])
    u_true = te["u"][i : i + 1]
    u_fno = preds_f[i]
    u_agf = preds_a[i]

    rel = lambda p: float(np.sqrt(((p - u_true) ** 2).sum())
                          / np.sqrt((u_true ** 2).sum()))
    print("per-sample FNO d16 rel-L2:",
          " ".join(f"{r:.3f}" for r in rels_f))
    print(f"chosen (median) sample {i} | FNO = {rel(u_fno):.4f} | "
          f"AGF-NO = {rel(u_agf):.4f}")

    x1, sdf1 = tensors(i)[0].cpu().numpy(), sdf[i : i + 1]
    out = os.path.join(ROOT, "paper", "figs", "hero_qualitative.png")
    U.make_comparison_figure(x1, sdf1, u_true, u_fno, u_agf, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
