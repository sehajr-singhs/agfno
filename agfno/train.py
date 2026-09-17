"""AGF-NO training orchestration (runs on Modal A100).

Local launcher (no GPU needed locally):

    python launch.py            # full pipeline
    python launch.py --quick    # short sanity run

Everything below `main` executes inside a Modal container with an A100:
data generation -> training (FNO + AGF-NO) -> evaluation -> figures ->
Hugging Face Hub upload. All artifacts are cached on a Modal volume, so
interrupted runs resume where they left off.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from . import config as C
from . import dataset as D
from . import utils as U
from .models import build_model


# --------------------------------------------------------------------------- #
# Tensor helpers
# --------------------------------------------------------------------------- #
def to_tensors(d: dict, device: str) -> dict:
    """Split arrays -> device tensors.

    Builds the two multi-channel stacks the models expect:
      x = [log K, sdf, ring]   (model input, in_ch = 3)
      g = [sdf, ring, interior] (geometry stream for AGF blocks, sdf_ch = 3)
    """
    t = {}
    for k in ("a", "u", "sdf", "interior", "ring"):
        t[k] = torch.tensor(d[k], dtype=torch.float32, device=device)
    t["x"] = torch.cat([t["a"], t["sdf"], t["ring"]], dim=1)
    t["g"] = torch.cat([t["sdf"], t["ring"], t["interior"]], dim=1)
    return t


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_one(
    name: str,
    data: dict,
    cfg_model,
    cfg_train,
    device: str,
    run_dir: str,
    quick: bool = False,
    seed: int = C.SEED,
    use_boundary_penalty: bool = True,
    n_blocks: int | None = None,
    gate_mode: str = "full",
    model_builder=None,
) -> tuple[torch.nn.Module, list]:
    """Train one operator. Returns (best model, history).

    ``seed``           -> model init + data-shuffle RNG (multi-seed stats),
    ``use_boundary_penalty`` -> toggle the boundary-loss term (loss ablation),
    ``n_blocks``       -> override depth (geometric-forgetting depth sweep),
    ``gate_mode``      -> AGF-NO control: "full" | "spec_only" | "frozen",
    ``model_builder``  -> (name, cfg_model, gate_mode) -> nn.Module. Defaults to
                          the 2-D ``build_model``; pass ``build_model3d`` for
                          3-D experiments (tensors are then [B, C, D, H, W]).
    """
    U.set_seed(seed)
    if n_blocks is not None:
        from dataclasses import replace

        cfg_model = replace(cfg_model, n_blocks=n_blocks)
    builder = model_builder or build_model
    model = builder(name, cfg_model, gate_mode=gate_mode).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train:{name}] parameters: {n_params/1e6:.2f}M")

    tr, va, te = data["train"], data["val"], data["test"]
    n_train = tr["a"].shape[0]
    epochs = 3 if quick else cfg_train.num_epochs
    bs = cfg_train.batch_size
    steps_per_epoch = max(1, n_train // bs)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg_train.lr,
                            weight_decay=cfg_train.weight_decay)
    warmup = max(1, int(cfg_train.warmup_frac * epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: min(1.0, (step + 1) / warmup)
        * 0.5
        * (1 + np.cos(np.pi * min(1.0, step / max(1, epochs * steps_per_epoch - warmup)))),
    )

    history = []
    best_val = float("inf")
    best_path = os.path.join(run_dir, f"{name}_best.pt")
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    step = 0
    last_good = None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        ep_loss, nb = 0.0, 0
        for i0 in range(0, n_train - bs + 1, bs):
            idx = perm[i0 : i0 + bs]
            a = tr["x"][idx]  # [B, 3, H, W] = [log K, sdf, ring]
            sdf = tr["g"][idx]  # [B, 3, H, W] geometry stream
            u = tr["u"][idx]
            itl = tr["interior"][idx]
            rg = tr["ring"][idx]

            pred = model(a, sdf)
            loss = U.relative_l2_loss(pred, u)
            total = loss
            if use_boundary_penalty:
                bnd = U.boundary_loss(pred, u, itl, rg)
                total = loss + cfg_train.boundary_lmbda * bnd

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
            opt.step()
            sched.step()
            step += 1
            ep_loss += total.item()
            nb += 1

            # Instability guard (deep nets at depth >= 8 can transiently
            # produce inf/NaN activations early in training): if the loss is
            # non-finite, restore the last good weights and halve the LR.
            if not torch.isfinite(total):
                print(f"[train:{name}] non-finite loss at epoch {epoch} step {i0 // bs}; "
                      f"restoring last good weights and halving LR")
                opt.param_groups[0]["lr"] *= 0.5
                if last_good is not None:
                    model.load_state_dict(last_good)
                opt.zero_grad(set_to_none=True)
                continue

        if epoch % cfg_train.eval_every == 0 or epoch == epochs or quick:
            m = U.eval_metrics(
                model, va["x"], va["g"], va["u"], va["interior"], va["ring"]
            )
            history.append(
                dict(epoch=epoch, train_loss=ep_loss / max(nb, 1),
                     val_rel_l2=m["rel_l2"], val_rel_l2_ring=m["rel_l2_ring"],
                     val_wall_viol=m["wall_viol"])
            )
            print(
                f"[train:{name}] epoch {epoch:4d}  loss {ep_loss/max(nb,1):.4f}  "
                f"val relL2 {m['rel_l2']:.4f}  ring {m['rel_l2_ring']:.4f}  "
                f"wall {m['wall_viol']:.4f}"
            )
            if m["rel_l2"] < best_val:
                best_val = m["rel_l2"]
                torch.save(model.state_dict(), best_path)
        # snapshot for the instability guard (after each clean epoch)
        last_good = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # restore best weights
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, history


# --------------------------------------------------------------------------- #
# Evaluation suite
# --------------------------------------------------------------------------- #
def evaluate_all(models: dict, data: dict, device: str, quick: bool) -> dict:
    """Test-split metrics + zero-shot super-resolution for every model."""
    results = {}
    te = data["test"]
    for name, model in models.items():
        m = U.eval_metrics(
            model, te["x"], te["g"], te["u"], te["interior"], te["ring"]
        )
        sr = {}
        if not quick and "test_hi" in data:
            sr = U.super_res_eval(model, data["test_hi"]["x"], data["test_hi"]["g"],
                                  data["test_hi"]["u"], device)
        results[name] = {**m, **sr}
        print(f"[eval:{name}] {results[name]}")
    return results


# --------------------------------------------------------------------------- #
# Hugging Face Hub upload
# --------------------------------------------------------------------------- #
def push_to_hf(run_dir: str, repo_id: str) -> str:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=run_dir,
        path_in_repo=".",
    )
    print(f"[hf] uploaded artifacts to https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


# --------------------------------------------------------------------------- #
# Main pipeline (executes inside Modal)
# --------------------------------------------------------------------------- #
def main(quick: bool = False, push: bool = True, artifacts_root: str | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] device = {device}")

    vol_root = C.INFRA.volume_mount if os.path.ismount(C.INFRA.volume_mount) or os.path.isdir(C.INFRA.volume_mount) else "/tmp/agfno_vol"
    os.makedirs(vol_root, exist_ok=True)
    # Artifacts (checkpoints/figures/results) can live separately from the
    # data cache -- e.g. on Kaggle, data in /kaggle/tmp, outputs in /kaggle/working.
    arts_root = artifacts_root or vol_root

    # ---------------- Data ----------------
    res_hi = C.DATA.res * 2
    splits = {}
    for split in ("train", "val", "test"):
        n = getattr(C.DATA, f"n_{split}")
        if quick:
            n = min(n, 256 if split == "train" else 64)
        splits[split] = D.load_or_make_split(split, n, C.DATA.res, vol_root, device)
    if not quick:
        splits["test_hi"] = D.load_or_make_split(
            "test", C.DATA.n_test, res_hi, vol_root, device
        )

    train_n, val_n, stats = D.normalize_split(splits["train"], splits["val"])
    _, test_n, _ = D.normalize_split(splits["train"], splits["test"])
    test_hi_n = None
    if "test_hi" in splits:
        _, test_hi_n, _ = D.normalize_split(splits["train"], splits["test_hi"])

    data = {
        "train": to_tensors(train_n, device),
        "val": to_tensors(val_n, device),
        "test": to_tensors(test_n, device),
    }
    if test_hi_n is not None:
        data["test_hi"] = to_tensors(test_hi_n, device)

    # ---------------- Training ----------------
    run_tag = "quick" if quick else "full"
    run_dir = os.path.join(arts_root, f"runs/{run_tag}_s{C.SEED}")
    os.makedirs(run_dir, exist_ok=True)

    models, histories = {}, {}
    for name in ("fno", "agfno"):
        t0 = time.time()
        model, hist = train_one(name, data, C.MODEL, C.TRAIN, device, run_dir, quick)
        dt = time.time() - t0
        models[name] = model
        histories[name] = hist
        print(f"[main] {name} trained in {dt/60:.1f} min")
    torch.cuda.synchronize() if device == "cuda" else None

    # Timing comparison (inference throughput, samples/sec)
    timing = {}
    te = data["test"]
    for name, model in models.items():
        model.eval()
        # warmup
        with torch.no_grad():
            for _ in range(3):
                model(te["x"][:32], te["g"][:32])
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(5):
                model(te["x"][:256], te["g"][:256])
        if device == "cuda":
            torch.cuda.synchronize()
        timing[name] = 256 * 5 / (time.time() - t0)
    print(f"[main] inference samples/sec: {timing}")

    # ---------------- Evaluation ----------------
    results = evaluate_all(models, data, device, quick)
    results["_timing_sps"] = timing

    # ---------------- Figures ----------------
    with torch.no_grad():
        a0 = te["x"][:1]
        sdf0 = te["g"][:1]
        u0 = te["u"][:1].cpu().numpy()
        preds = {n: models[n](a0, sdf0).cpu().numpy() for n in models}
    U.make_comparison_figure(
        a0.cpu().numpy(), te["sdf"][:1].cpu().numpy(), u0,
        preds["fno"], preds["agfno"],
        os.path.join(run_dir, "comparison.png"),
    )
    U.make_error_history_figure(
        histories, os.path.join(run_dir, "training_curves.png")
    )
    if not quick and "test_hi" in data:
        sr_plot = {
            n: dict(rel_l2=results[n]["rel_l2"], rel_l2_sr=results[n].get("rel_l2_sr"))
            for n in models
        }
        U.make_sr_figure(sr_plot, os.path.join(run_dir, "super_resolution.png"))

    # ---------------- Save everything ----------------
    meta = dict(
        seed=C.SEED,
        device=device,
        gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        data_config={k: str(v) for k, v in C.DATA.__dict__.items() if not k.startswith("_")},
        model_config={k: str(v) for k, v in C.MODEL.__dict__.items() if not k.startswith("_")},
        train_config={k: str(v) for k, v in C.TRAIN.__dict__.items() if not k.startswith("_")},
        normalization={k: float(v) for k, v in stats.items()},
        timing_samples_per_sec=timing,
        results=results,
        histories=histories,
    )
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for name in models:
        torch.save(models[name].state_dict(), os.path.join(run_dir, f"{name}_final.pt"))

    # ---------------- HF Hub ----------------
    url = None
    if push:
        repo = os.environ.get("AGFNO_HF_REPO", "").strip()
        if repo:
            try:
                url = push_to_hf(run_dir, repo)
            except Exception as e:
                print(f"[hf] upload failed (non-fatal): {e}")
        else:
            print("[hf] AGFNO_HF_REPO not set; skipping upload")

    print(json.dumps({k: v for k, v in results.items()}, indent=2, default=str))
    return {"run_dir": run_dir, "results": results, "hf_url": url}
