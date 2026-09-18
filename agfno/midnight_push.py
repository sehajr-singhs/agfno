#!/usr/bin/env python
"""The headless horseman: wait for the Kaggle GPU quota reset, then ride.

Fires the two staged GPU suites the moment the weekly quota resets
(Sat 00:00 UTC), polls to completion, pulls the outputs, merges, and
rebuilds the paper. Designed to run DETACHED -- no interaction, all state
in midnight_state.json, every action logged to midnight.log:

    nohup python agfno/midnight_push.py > /dev/null 2>&1 &

Sequence:
  1. wait until the GPU quota accepts a push (probe by pushing exp5);
  2. push the three kernels: exp5 (full-budget baselines), exp6-afno,
     exp6-agfafno (two concurrent sessions, one per arm);
  3. poll all three every 5 min until each is COMPLETE or ERROR;
  4. pull outputs into kaggle/exp5_out and kaggle/exp6_out/{afno,agfafno};
  5. merge exp6 arms (merge_exp6), rebuild macros + PDF;
  6. write final status to midnight_state.json.

Everything is idempotent: if killed and relaunched, completed stages are
skipped based on state, and kernels are re-pushed only if never accepted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "midnight_state.json"
LOG = ROOT / "midnight.log"
KDIR = ROOT / "kaggle" / "kernels"

KERNELS = [
    ("exp5", KDIR / "exp5", "sehajrsingh/agfno-darcy-experiments5"),
    ("exp6-afno", KDIR / "exp6-afno", "sehajrsingh/agfno-darcy-exp6-afno"),
    ("exp6-agfafno", KDIR / "exp6-agfafno", "sehajrsingh/agfno-darcy-exp6-agfafno"),
]


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"pushed": {}, "status": {}, "pulled": {}, "merged": False}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2))


def kaggle(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    """Run a kaggle CLI call; quota errors are expected, not crashes."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "kaggle", *args],
                       capture_output=True, text=True, timeout=900, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {out[-500:]}")
    return r


def push(kernel_dir: Path, slug: str) -> bool:
    """Push one kernel; True on acceptance (or already-running/complete)."""
    r = kaggle("kernels", "push", "-p", str(kernel_dir))
    out = (r.stdout or "") + (r.stderr or "")
    if "Successfully pushed" in out or "kernel-metadata.json successfully" in out:
        log(f"pushed {slug}")
        return True
    if "quota" in out.lower() and "gpu" in out.lower():
        log(f"quota still exhausted ({slug})")
        return False
    if "has an active running session" in out or "already" in out.lower():
        log(f"{slug} already has a session")
        return True
    log(f"push {slug} unexpected: {out[-300:]}")
    return False


def status(slug: str) -> str:
    r = kaggle("kernels", "status", slug)
    out = ((r.stdout or "") + (r.stderr or "")).lower()
    if "complete" in out:
        return "COMPLETE"
    if "error" in out or "cancel" in out:
        return "ERROR"
    if "running" in out or "queued" in out:
        return "RUNNING"
    if "404" in out or "not found" in out:
        return "MISSING"
    return out.strip()[-120:] or "UNKNOWN"


def pull(slug: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    r = kaggle("kernels", "output", slug, "-p", str(dest))
    out = (r.stdout or "") + (r.stderr or "")
    ok = r.returncode == 0 and any(dest.glob("*"))
    log(f"pull {slug} -> {dest.name}: {'OK' if ok else 'FAIL ' + out[-200:]}")
    return ok


def rebuild_paper() -> bool:
    """merge exp6 arms -> macros -> 2x pdflatex. Returns success."""
    log("merging exp6 arms and rebuilding paper")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    m = subprocess.run(
        [sys.executable, "-m", "agfno.merge_exp6",
         str(ROOT / "kaggle/exp6_out/afno"),
         str(ROOT / "kaggle/exp6_out/agfafno"),
         "--out", str(ROOT / "kaggle/exp6_out")],
        cwd=ROOT, capture_output=True, text=True, env=env)
    if m.returncode != 0:
        log(f"merge_exp6 FAILED: {(m.stderr or m.stdout)[-400:]}")
        return False
    log("merge_exp6 OK; gates:\n" + (m.stdout or "")[-600:])
    g = subprocess.run([sys.executable, "paper/gen_paper.py"], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    if g.returncode != 0:
        log(f"gen_paper FAILED: {(g.stderr or g.stdout)[-400:]}")
        return False
    for _ in range(2):
        c = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-halt-on-error", "main.tex"], cwd=ROOT / "paper",
                           capture_output=True, text=True)
        if c.returncode != 0:
            log("pdflatex FAILED")
            return False
    log("paper rebuilt OK")
    return True


def main() -> None:
    s = load_state()
    log(f"headless horseman mounts. state: pushed={s['pushed']} status={s['status']}")

    # ---- stage 1-2: push everything (retries until quota accepts) --------- #
    deadline = time.time() + 3600 * 3  # give up after 3 h of quota-waiting
    while time.time() < deadline:
        pending = [k for k in KERNELS if not s["pushed"].get(k[0])]
        if not pending:
            break
        for key, kdir, slug in pending:
            s["pushed"][key] = push(kdir, slug)
            save_state(s)
            if not s["pushed"][key] and "quota" in (status(slug) or ""):
                pass  # quota response comes from push(); just wait
        if all(s["pushed"].values()):
            break
        log("sleeping 600s before next push attempt")
        time.sleep(600)
    if not all(s["pushed"].values()):
        log("GIVING UP: quota did not reset within the deadline; relaunch later.")
        save_state(s)
        return
    log("all kernels accepted")

    # ---- stage 3: poll ----------------------------------------------------- #
    while True:
        done = True
        for key, _kdir, slug in KERNELS:
            if s["status"].get(key) in ("COMPLETE", "ERROR"):
                continue
            st = status(slug)
            log(f"status {key}: {st}")
            s["status"][key] = st
            save_state(s)
            if st not in ("COMPLETE", "ERROR"):
                done = False
        if done:
            break
        time.sleep(300)
    log(f"all sessions ended: {s['status']}")

    # ---- stage 4: pull ----------------------------------------------------- #
    pulls = [
        ("exp5", KERNELS[0][2], ROOT / "kaggle/exp5_out"),
        ("exp6-afno", KERNELS[1][2], ROOT / "kaggle/exp6_out/afno"),
        ("exp6-agfafno", KERNELS[2][2], ROOT / "kaggle/exp6_out/agfafno"),
    ]
    for key, slug, dest in pulls:
        if s["status"].get(key) == "COMPLETE" and not s["pulled"].get(key):
            s["pulled"][key] = pull(slug, dest)
            save_state(s)

    # ---- stage 5-6: merge + rebuild ---------------------------------------- #
    if s["pulled"].get("exp6-afno") and s["pulled"].get("exp6-agfafno"):
        s["merged"] = rebuild_paper()
        save_state(s)
    else:
        log("exp6 outputs missing; skipping merge (exp5 may still be usable)")
    log(f"horseman dismounts. final state: {json.dumps(s, indent=1)}")


if __name__ == "__main__":
    main()
