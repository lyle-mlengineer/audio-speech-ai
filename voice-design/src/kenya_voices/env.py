"""Colab environment checks + provenance: GPU, Drive, VoxCPM2 repo, base model.

Facts this module encodes (verified against upstream code/docs):

* The trainer autocasts to **bfloat16**, so Turing (T4, no native bf16) emulates
  bf16 and trains several times slower. The official VRAM figure is ~20 GB for
  VoxCPM2 LoRA, i.e. an L4 (24 GB) or A100. T4 is allowed only via
  ``allow_small=True`` with a reduced config.
* ``pip install -e .`` on the VoxCPM repo tries to pull ``torch>=2.5``,
  ``funasr``, ``modelscope``, ``gradio``... which can silently replace Colab's
  CUDA-matched torch. Default install is therefore ``--no-deps`` plus a curated
  list; ``full_deps=True`` opts into the plain official install.
* ``tensorboardX`` is imported by the training script but is *not* in VoxCPM's
  dependency list, so it must be installed explicitly.
* The setup marker is kept in the VM-local cache, never on Drive: a Drive marker
  would make a *fresh* Colab VM think the packages are already installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import BASE_MODEL_ID, VOXCPM_REPO_URL, Paths

SETUP_MARKER_NAME = "setup.json"

MIN_VRAM_GB = 20.0
MIN_CC_FOR_BF16 = 8.0  # Ampere (A100 / L4 / A10 / 30-40 series) and newer

# package name -> import name (for a cheap "already installed?" check)
CURATED_DEPS: dict[str, str] = {
    "argbind": "argbind",
    "transformers>=4.36.2": "transformers",
    "safetensors": "safetensors",
    "einops": "einops",
    "addict": "addict",
    "simplejson": "simplejson",
    "sortedcontainers": "sortedcontainers",
    "soundfile": "soundfile",
    "inflect": "inflect",
    "datasets>=3,<4": "datasets",
    "huggingface-hub": "huggingface_hub",
    "pydantic": "pydantic",
    "librosa": "librosa",
    "matplotlib": "matplotlib",
    "tensorboardX": "tensorboardX",
    "pyyaml": "yaml",
    "scipy": "scipy",
}

REQUIRED_IMPORTS = ("torch", "torchaudio", "datasets", "soundfile", "librosa", "transformers")


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


def marker_path() -> Path:
    """VM-local setup marker (never on Drive - see module docstring)."""
    base = Path.home() / ".cache" / "kenya-voices"
    base.mkdir(parents=True, exist_ok=True)
    return base / SETUP_MARKER_NAME


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
        return dict(self.__dict__)

    def summary(self) -> str:
        if not self.available:
            return f"no CUDA GPU | torch {self.torch}"
        return f"{self.name} | {self.vram_gb:.1f} GB | cc {self.cc:.1f} | native bf16 {self.bf16} | torch {self.torch}"

    @property
    def profile(self) -> str:
        """Best training profile for this GPU (see configs/)."""
        if not self.available:
            return "t4"
        if self.vram_gb >= 70:
            return "a100"
        if self.vram_gb >= 22 and self.bf16:
            return "l4"
        return "t4"


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
            "(L4 recommended, A100 if you have Pro+/Pay-as-you-go; T4 works only with --allow-small-gpu)."
        )
    problems = []
    if not info.bf16:
        problems.append(f"{info.name} has compute capability {info.cc:.1f} < 8.0 (no native bf16)")
    if info.vram_gb + 0.01 < MIN_VRAM_GB:
        problems.append(f"{info.vram_gb:.1f} GB VRAM < the ~{MIN_VRAM_GB:.0f} GB the LoRA config needs")
    if problems and not allow_small:
        raise RuntimeError(
            "GPU is not ideal for VoxCPM2 LoRA fine-tuning:\n  - "
            + "\n  - ".join(problems)
            + "\nUse an L4 (Colab Pro) or A100. To try anyway with the reduced "
            "'t4' profile use --allow-small-gpu (expect slow bf16 emulation; enable "
            "smaller batch/grad settings automatically)."
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
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _missing_imports() -> list[str]:
    missing = []
    for module in REQUIRED_IMPORTS:
        try:
            __import__(module)
        except Exception:  # noqa: BLE001 - broken install counts as missing
            missing.append(module)
    return missing


def missing_imports() -> list[str]:
    """Public wrapper used by ``kenya-voices doctor``."""
    return _missing_imports()


def missing_curated() -> list[str]:
    """Curated training deps that are not importable right now."""
    return [pkg for pkg, mod in CURATED_DEPS.items() if _module_missing(mod)]


def ensure_repo(
    paths: Paths,
    *,
    ref: str = "main",
    update: bool = False,
    full_deps: bool = False,
    reinstall: bool = False,
    log=print,
) -> dict[str, Any]:
    """Clone OpenBMB/VoxCPM at ``ref`` and pip-install it. Returns provenance."""
    repo = paths.voxcpm_repo
    train_script = paths.train_script

    if not train_script.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        if repo.exists():
            log(f"[env] {repo} exists but looks incomplete - recloning")
            import shutil

            shutil.rmtree(repo)
        # Full clone: the package version comes from setuptools_scm + git tags.
        run(["git", "clone", VOXCPM_REPO_URL, str(repo)], log=log)
    if update:
        run(["git", "-C", str(repo), "fetch", "--all", "--tags", "--quiet"], log=log, check=False)
    run(["git", "-C", str(repo), "checkout", ref, "--quiet"], log=log, check=False)
    rev = git_rev(repo)

    mode = "full" if full_deps else "minimal"
    try:
        installed = json.loads(marker_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        installed = {}
    missing = _missing_imports()

    needs_install = reinstall or missing or installed.get("rev") != rev or installed.get("mode") != mode
    if needs_install:
        log(f"[env] installing voxcpm from {repo} (mode={mode}, missing={missing or 'none'})")
        base = [sys.executable, "-m", "pip", "install", "-e", str(repo)]
        if not full_deps:
            base.insert(4, "--no-deps")
        run(base, log=log)
        to_install = [pkg for pkg, mod in CURATED_DEPS.items() if _module_missing(mod)]
        if to_install:
            run([sys.executable, "-m", "pip", "install", *to_install], log=log)
        if full_deps:
            run([sys.executable, "-m", "pip", "install", "tensorboardX"], log=log)
        marker_path().write_text(
            json.dumps({"rev": rev, "mode": mode, "ref": ref, "torch": _torch_version()}, indent=2),
            encoding="utf-8",
        )
    else:
        log(f"[env] voxcpm already installed at {rev} (mode={mode})")

    return {"repo": str(repo), "ref": ref, "rev": rev, "mode": mode, "torch": _torch_version()}


def _module_missing(module: str) -> bool:
    try:
        __import__(module)
        return False
    except Exception:  # noqa: BLE001
        return True


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "missing"


# ------------------------------------------------------------------- model
def _architecture(model_dir: Path) -> str | None:
    try:
        return json.loads((model_dir / "config.json").read_text(encoding="utf-8")).get("architecture")
    except (OSError, json.JSONDecodeError):
        return None


def model_architecture(model_dir: Path) -> str | None:
    """Public wrapper: ``config.json``'s ``architecture`` (``voxcpm2`` for our base)."""
    return _architecture(model_dir)


def ensure_model(paths: Paths, *, revision: str | None = None, log=print) -> Path:
    """Download openbmb/VoxCPM2 into a local dir (training needs a path, not an id)."""
    target = paths.model_dir
    arch = _architecture(target)
    if arch and arch.lower() == "voxcpm2":
        log(f"[model] reusing {target}")
        return target
    if arch:
        log(f"[model] {target} has architecture={arch!r} - re-downloading")

    from huggingface_hub import snapshot_download

    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"[model] downloading {BASE_MODEL_ID} -> {target} (several GB, resumable)")
    snapshot_download(
        repo_id=BASE_MODEL_ID,
        revision=revision,
        local_dir=str(target),
    )
    arch = _architecture(target)
    if not arch or arch.lower() != "voxcpm2":
        raise RuntimeError(f"downloaded model is architecture={arch!r}, expected voxcpm2 (dir: {target})")
    log("[model] ok: architecture=voxcpm2")
    return target
