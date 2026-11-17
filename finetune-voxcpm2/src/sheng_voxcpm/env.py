"""Colab environment checks + provenance: GPU, Drive, VoxCPM repo, base model.

Design notes that came out of the upstream code/docs:

* The trainer autocasts to **bfloat16**, so Turing (T4 / 16 GB, no bf16) is
  rejected by default - it trains slowly and can produce bad audio.
  The official VRAM figure is ~20 GB for VoxCPM2 LoRA -> L4 (24 GB) or A100.
* ``pip install -e .`` on the VoxCPM repo pulls ``torch>=2.5``, ``funasr``,
  ``gradio``, ``modelscope``... which can silently replace Colab's CUDA-matched
  torch. Default install mode is therefore ``--no-deps`` + a curated list;
  ``--full-deps`` opts into the plain official install.
* ``tensorboardX`` is imported by the training script but is *not* in VoxCPM's
  dependency list, so we install it explicitly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import BASE_MODEL_ID, VOXCPM_REPO_URL, Paths

SETUP_MARKER = ".sheng-setup.json"

MIN_VRAM_GB = 20.0
MIN_CC_FOR_BF16 = 8.0  # Ampere (A100 / L4 / A10 / 30/40-series) and newer

CURATED_DEPS = [
    "argbind",
    "transformers>=4.36.2",
    "safetensors",
    "einops",
    "addict",
    "simplejson",
    "sortedcontainers",
    "soundfile",
    "inflect",
    "datasets>=3,<4",
    "huggingface-hub",
    "pydantic",
    "librosa",
    "matplotlib",
    "tensorboardX",
    "pyyaml",
    "scipy",
]


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True, log=print) -> subprocess.CompletedProcess:
    log(f"$ {' '.join(cmd)}" + (f"   (cwd={cwd})" if cwd else ""))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    if proc.stdout.strip():
        log(proc.stdout.strip()[-4000:])
    if proc.returncode != 0:
        log(proc.stderr.strip()[-4000:])
        if check:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


# --------------------------------------------------------------------- GPU
@dataclass
class GpuInfo:
    available: bool
    name: str = "none"
    vram_gb: float = 0.0
    cc: float = 0.0
    bf16: bool = False
    torch: str = "n/a"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def summary(self) -> str:
        return f"{self.name} | {self.vram_gb:.1f} GB | cc {self.cc:.1f} | bf16 {self.bf16} | torch {self.torch}"


def gpu_info() -> GpuInfo:
    try:
        import torch
    except ImportError:
        return GpuInfo(available=False)
    if not torch.cuda.is_available():
        return GpuInfo(available=False, torch=torch.__version__)
    props = torch.cuda.get_device_properties(0)
    cc = float(f"{props.major}.{props.minor}")
    return GpuInfo(
        available=True,
        name=props.name,
        vram_gb=round(props.total_memory / 1024**3, 1),
        cc=cc,
        bf16=cc >= MIN_CC_FOR_BF16,
        torch=torch.__version__,
    )


def check_gpu(info: GpuInfo | None = None, *, allow_small: bool = False) -> GpuInfo:
    info = info or gpu_info()
    if not info.available:
        raise RuntimeError(
            "No CUDA GPU. In Colab: Runtime > Change runtime type > Hardware accelerator > GPU "
            "(L4 recommended, A100 if you have Pro+/Pay-as-you-go)."
        )
    problems = []
    if not info.bf16:
        problems.append(f"{info.name} has compute capability {info.cc:.1f} < 8.0 (no native bf16)")
    if info.vram_gb + 0.01 < MIN_VRAM_GB:
        problems.append(f"{info.vram_gb:.1f} GB VRAM < the ~{MIN_VRAM_GB:.0f} GB the official LoRA config needs")
    if problems and not allow_small:
        raise RuntimeError(
            "GPU is not suitable for VoxCPM2 LoRA fine-tuning:\n  - "
            + "\n  - ".join(problems)
            + "\nUse an L4 (Colab free/Pro) or A100. To try anyway with a reduced "
            "config use --allow-small-gpu (expect OOM or slow/bf16-emulated training)."
        )
    return info


# ------------------------------------------------------------------- Drive
def mount_drive(log=print) -> Path | None:
    """Mount Google Drive on Colab; returns the MyDrive path if available."""
    target = Path("/content/drive/MyDrive")
    if target.is_dir():
        return target
    try:
        from google.colab import drive  # type: ignore

        log("[env] mounting Google Drive (authorise the popup)")
        drive.mount("/content/drive")
        return target if target.is_dir() else None
    except Exception as exc:  # noqa: BLE001 - not on Colab / user declined
        log(f"[env] Drive not mounted ({exc.__class__.__name__}: {exc})")
        return None


# -------------------------------------------------------------------- repo
def git_rev(repo: Path) -> str:
    proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def ensure_repo(
    paths: Paths,
    *,
    ref: str = "main",
    update: bool = False,
    full_deps: bool = False,
    reinstall: bool = False,
    log=print,
) -> dict[str, Any]:
    """Clone ../VoxCPM at ``ref`` and pip-install it. Returns provenance info."""
    repo = paths.voxcpm_repo
    train_script = repo / "scripts" / "train_voxcpm_finetune.py"

    if not train_script.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        if repo.exists():
            log(f"[env] {repo} exists but looks incomplete — recloning")
            import shutil

            shutil.rmtree(repo)
        run(["git", "clone", VOXCPM_REPO_URL, str(repo)], log=log)
    if update:
        run(["git", "-C", str(repo), "fetch", "--all", "--tags", "--quiet"], log=log, check=False)
    run(["git", "-C", str(repo), "checkout", ref, "--quiet"], log=log, check=False)
    rev = git_rev(repo)

    marker = repo / SETUP_MARKER
    mode = "full" if full_deps else "minimal"
    installed = None
    if marker.exists():
        try:
            installed = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            installed = None

    needs_install = reinstall or not installed or installed.get("rev") != rev or installed.get("mode") != mode
    if needs_install:
        log(f"[env] installing voxcpm from {repo} (mode={mode})")
        base = [sys.executable, "-m", "pip", "install", "-e", str(repo)]
        if not full_deps:
            base.insert(4, "--no-deps")
        run(base, log=log)
        run([sys.executable, "-m", "pip", "install", *CURATED_DEPS], log=log)
        if full_deps:
            run([sys.executable, "-m", "pip", "install", "tensorboardX"], log=log)
        marker.write_text(
            json.dumps({"rev": rev, "mode": mode, "ref": ref, "torch": _torch_version()}, indent=2),
            encoding="utf-8",
        )
    else:
        log(f"[env] voxcpm already installed at {rev} (mode={mode})")

    return {"repo": str(repo), "ref": ref, "rev": rev, "mode": mode, "torch": _torch_version()}


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "missing"


# ------------------------------------------------------------------- model
def ensure_model(paths: Paths, *, revision: str | None = None, log=print) -> Path:
    """Download openbmb/VoxCPM2 into a local dir (training needs a path, not an id)."""
    target = paths.model_dir
    cfg = paths.base_config_json
    if cfg.exists():
        try:
            arch = json.loads(cfg.read_text(encoding="utf-8")).get("architecture", "voxcpm").lower()
        except json.JSONDecodeError:
            arch = "unreadable"
        if arch == "voxcpm2":
            log(f"[model] reusing {target}")
            return target
        log(f"[model] {target} exists but architecture={arch!r} — re-downloading")

    from huggingface_hub import snapshot_download

    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"[model] downloading {BASE_MODEL_ID} -> {target} (several GB, resumable)")
    snapshot_download(
        repo_id=BASE_MODEL_ID,
        revision=revision,
        local_dir=str(target),
        resume_download=True,
        max_workers=8,
    )
    arch = json.loads(cfg.read_text(encoding="utf-8")).get("architecture", "voxcpm").lower()
    if arch != "voxcpm2":
        raise RuntimeError(f"downloaded model is architecture={arch!r}, expected voxcpm2 (dir: {target})")
    log("[model] ok: architecture=voxcpm2")
    return target
