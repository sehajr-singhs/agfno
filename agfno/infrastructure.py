"""Modal infrastructure: container image, secrets, volumes, remote jobs.

Usage (from the project root, after `modal token new`):

    modal run agfno/infrastructure.py::smoke        # CPU smoke test
    modal run agfno/infrastructure.py::envcheck     # verify image + secrets
    modal run agfno/infrastructure.py::selftest     # full GPU test suite
"""

from __future__ import annotations

import modal

from . import config as C

app = modal.App(name=C.INFRA.app_name)

# Secrets mounted into every job: Kaggle credentials + Hugging Face token.
# Create once with:
#   modal secret create agfno-hf HF_TOKEN=hf_xxx
#   modal secret create agfno-kaggle KAGGLE_USERNAME=... KAGGLE_KEY=...
def _get_secret(name: str):
    """Attach a Modal secret if it exists; warn (don't crash) otherwise."""
    try:
        return modal.Secret.from_name(name)
    except Exception as e:  # secret not found at app-build time
        print(f"[infra] Modal secret '{name}' not found ({e}); continuing without it")
        return None


secrets = [s for s in (_get_secret("agfno-hf"), _get_secret("agfno-kaggle")) if s]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "numpy<2.3",
        "scipy",
        "matplotlib",
        "huggingface_hub",
        "kaggle",
        "kagglehub",
        "tqdm",
        "filelock",
        "pytest",
    )
    .add_local_python_source("agfno")
)

volume = modal.Volume.from_name(C.INFRA.volume_name, create_if_missing=True)


@app.function(image=image, secrets=secrets, timeout=300)
def envcheck() -> dict:
    """Verify the container: torch, CUDA availability, secrets, hub access."""
    import os

    import torch

    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "HF_TOKEN_set": bool(os.environ.get("HF_TOKEN")),
        "KAGGLE_KEY_set": bool(os.environ.get("KAGGLE_KEY")),
        "AGFNO_HF_REPO": os.environ.get("AGFNO_HF_REPO", "<unset>"),
    }
    try:
        from huggingface_hub import HfApi

        who = HfApi(token=os.environ.get("HF_TOKEN")).whoami()
        info["hf_user"] = who.get("name", "<unknown>")
    except Exception as e:
        info["hf_user"] = f"<error: {e}>"
    print(info)
    return info


@app.function(
    image=image,
    secrets=secrets,
    timeout=1800,
    volumes={C.INFRA.volume_mount: volume},
    cpu=C.INFRA.cpu_cores,
)
def smoke() -> dict:
    """CPU-only smoke test of the data + model + training step."""
    import torch

    from . import dataset as D
    from . import utils as U
    from .models import build_model

    U.set_seed(C.SEED)
    d = D.make_split("val", n=8, res=C.DATA.res, device="cpu")
    assert d["a"].shape == (8, 1, C.DATA.res, C.DATA.res), d["a"].shape
    assert d["u"].shape == d["a"].shape
    assert d["sdf"].shape == d["a"].shape

    cfg = C.MODEL
    for name in ("fno", "agfno"):
        m = build_model(name, cfg)
        a = torch.tensor(d["a"][:2], dtype=torch.float32)
        sdf = torch.tensor(d["sdf"][:2], dtype=torch.float32)
        out = m(a, sdf)
        assert out.shape == (2, 1, C.DATA.res, C.DATA.res)
        loss = U.relative_l2_loss(out, torch.tensor(d["u"][:2]))
        loss.backward()
    return {"smoke": "ok", "res": C.DATA.res, "device": "cpu"}


@app.function(
    image=image,
    secrets=secrets,
    timeout=1800,
    gpu=C.INFRA.gpu,
    volumes={C.INFRA.volume_mount: volume},
)
def selftest() -> dict:
    """Run the pytest suite on an A100 GPU."""
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "agfno/tests", "-v", "--tb=short", "-x"],
        capture_output=True,
        text=True,
    )
    print(r.stdout)
    print(r.stderr)
    return {"returncode": r.returncode, "passed": r.returncode == 0}
