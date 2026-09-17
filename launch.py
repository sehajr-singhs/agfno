#!/usr/bin/env python
"""Launcher for the AGF-NO pipeline on Modal.

Examples
--------
    python launch.py --envcheck          # verify image, GPU, secrets
    python launch.py --smoke             # CPU smoke test
    python launch.py --test              # pytest suite on an A100
    python launch.py --quick             # short end-to-end sanity run
    python launch.py                     # full research run (A100)
    python launch.py --data-only         # just generate/cache the dataset
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def sh(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="AGF-NO on Modal")
    p.add_argument("--envcheck", action="store_true", help="verify env + secrets")
    p.add_argument("--smoke", action="store_true", help="CPU smoke test")
    p.add_argument("--test", action="store_true", help="run pytest on A100")
    p.add_argument("--quick", action="store_true", help="quick end-to-end run")
    p.add_argument("--data-only", action="store_true", help="generate data only")
    p.add_argument("--no-push", action="store_true", help="skip HF Hub upload")
    p.add_argument("--repo", type=str, default=None,
                   help="Hugging Face repo id (overrides AGFNO_HF_REPO)")
    args = p.parse_args()

    base = ["modal", "run", "agfno/infrastructure.py"]
    if args.envcheck:
        return sh(base + ["::envcheck"])
    if args.smoke:
        return sh(base + ["::smoke"])
    if args.test:
        return sh(base + ["::selftest"])
    if args.data_only:
        return sh(base + ["::make_data"])

    flags = ["--quick"] if args.quick else []
    if args.no_push:
        flags.append("--no-push")
    if args.repo:
        flags += ["--hf-repo", args.repo]
    return sh(["modal", "run", "agfno/run.py"] + flags)


if __name__ == "__main__":
    sys.exit(main())
