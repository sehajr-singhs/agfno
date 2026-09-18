#!/usr/bin/env python
"""Fetch exp5 (matched-baselines) artifacts from the Modal volume, then
rebuild macros + PDF. The one command that closes the baselines loop once
GPU access returns:

    modal run -m agfno.infrastructure::fetch_exp5 > raw.log   (any log target)
    python agfno/fetch_exp5.py raw.log                        (parse + rebuild)

The parser greps ``EXP5JSON::<rel>::<json>`` marker lines out of the raw
output (ignoring Modal's banners/warnings), writes the files under
kaggle/exp5_out/runs/experiments5/, then regenerates paper/macros.tex and
recompiles the PDF -- which auto-inserts the baselines section the moment
\\UNetRelL is defined.

Exit codes: 0 = artifacts landed and PDF rebuilt; 1 = no artifacts found
(job probably not finished yet -- re-run ``modal run -m
agfno.infrastructure::fetch_exp5`` later).
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent          # agfno/
ROOT = HERE.parent                                       # project root
OUT = ROOT / "kaggle" / "exp5_out" / "runs" / "experiments5"
PAPER = ROOT / "paper"

MARKER = re.compile(r"EXP5JSON::(.+?)::(\{.*)$")


def parse_log(log_path: pathlib.Path) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = MARKER.search(raw)
        if not m:
            continue
        rel, payload = m.group(1).strip(), m.group(2).strip()
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as e:
            print(f"  [warn] {rel}: bad JSON ({e})")
            continue
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(f"  [ok] {rel}")
        n += 1
    return n


def rebuild_paper() -> bool:
    print("[fetch_exp5] regenerating macros ...")
    r = subprocess.run([sys.executable, str(PAPER / "gen_paper.py")],
                       capture_output=True, text=True)
    print(r.stdout[-1500:] if r.stdout else "", r.stderr[-500:] if r.returncode else "")
    if r.returncode:
        return False
    macros = (PAPER / "macros.tex").read_text(encoding="utf-8")
    if "\\UNetRelL}{??" in macros:
        print("[fetch_exp5] macros still PENDING -- not rebuilding PDF")
        return False
    print("[fetch_exp5] rebuilding PDF (2 passes) ...")
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=PAPER, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-1200:])
            return False
    print(f"[fetch_exp5] PDF rebuilt: {PAPER / 'main.pdf'}")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    log_path = pathlib.Path(sys.argv[1])
    if not log_path.exists():
        print(f"log not found: {log_path}")
        return 2
    n = parse_log(log_path)
    if n == 0:
        print("[fetch_exp5] no EXP5JSON artifacts in log -- run not finished?")
        return 1
    print(f"[fetch_exp5] {n} artifact file(s) written to {OUT}")
    ok = rebuild_paper()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
