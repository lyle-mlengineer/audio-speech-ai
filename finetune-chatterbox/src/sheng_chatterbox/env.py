"""Environment: GPU checks, package install, Drive mount, model download, doctor.

Everything here is deliberately defensive - it runs on a fresh Colab VM where
torch may or may not be installed yet, and it must never be the reason a run dies.

``chatterbox-tts`` pins ``torch==2.6.0`` / ``torchaudio==2.6.0``. Installing it
with its own dependencies therefore *replaces* the CUDA build Colab shipped with
and invalidates the running kernel. We install it with ``--no-deps`` and pull in
only the packages it imports at inference/training time, so the Colab torch (and
its CUDA wheels) stay untouched. ``--full-deps`` opts into the upstream behaviour.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .paths import (
    MTL_T3_MODEL,
    PRETRAINED_FILES,
    SPEAKER,
    MODEL_REPO_ID,
    Paths,
    colab_drive_mounted,
    on_drive,
)

CHATTERBOX_VERSION = "0.1.7"

# Imported by chatterbox / our trainer; deliberately excludes torch, torchaudio,
# torchvision and gradio (Colab ships torch, we never serve a UI).
RUNTIME_PACKAGES = [
    "librosa==0.11.0",
    "s3tokenizer",
    "diffusers",
    "conformer==0.3.2",
    "safetensors>=0.4.0",
    "resemble-perth>=1.0.0",
    "pyloudnorm",
    "omegaconf",
    "tokenizers",
    "transformers>=4.46",
    "datasets>=2.19",
    "soundfile",
    "tensorboard",
    "tqdm",
]

VM_MARKER = Path(os.environ.get("SHENG_CHATTERBOX_VM_MARKER", "/tmp/sheng_chatterbox_env.json"))


def run(cmd: list[str], *, log=print, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout.strip():
        log(proc.stdout.strip()[-4000:])
    if proc.returncode != 0:
        log((proc.stderr or "").strip()[-4000:])
        if check:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


# ------------------------------------------------------------------- GPU info
def gpu_info() -> dict[str, Any]:
    """Best-effort GPU facts. Safe with no torch installed."""
    info: dict[str, Any] = {"cuda": False, "name": "cpu", "vram_gb": 0.0, "bf16": False}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"torch not importable: {exc}"
        return info
    try:
        if not torch.cuda.is_available():
            return info
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "cuda": True,
                "name": props.name,
                "vram_gb": round(props.total_memory / 2**30, 1),
                "cc": f"{props.major}.{props.minor}",
                "bf16": bool(torch.cuda.is_bf16_supported()),
                "torch": torch.__version__,
            }
        )
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


# -------------------------------------------------------------------- install
def probe_imports() -> dict[str, str | None]:
    """Import-test the packages the trainer needs; returns {module: version|None}."""
    probes = {
        "torch": "torch",
        "torchaudio": "torchaudio",
        "chatterbox": "chatterbox",
        "transformers": "transformers",
        "datasets": "datasets",
        "soundfile": "soundfile",
        "safetensors": "safetensors",
        "librosa": "librosa",
        "s3tokenizer": "s3tokenizer",
    }
    out: dict[str, str | None] = {}
    for name, mod in probes.items():
        try:
            module = __import__(mod)
            out[name] = getattr(module, "__version__", "ok")
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def install(
    *,
    force: bool = False,
    full_deps: bool = False,
    log=print,
) -> dict[str, Any]:
    """pip-install chatterbox-tts (+ the deps we need). VM-local, not on Drive."""
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    pip = [sys.executable, "-m", "pip", "install", "-q"]

    if full_deps:
        log("[env] full upstream install (this REPLACES Colab's torch - restart the runtime after)")
        run(pip + [f"chatterbox-tts=={CHATTERBOX_VERSION}"], log=log)
    else:
        log(f"[env] chatterbox-tts=={CHATTERBOX_VERSION} --no-deps (keeps Colab's CUDA torch)")
        run(pip + ["--no-deps", f"chatterbox-tts=={CHATTERBOX_VERSION}"], log=log)
        run(pip + RUNTIME_PACKAGES, log=log)

    probes = probe_imports()
    missing = [name for name, version in probes.items() if version is None]
    marker = {"installed_at": started, "full_deps": bool(full_deps), "probes": probes, "missing": missing}
    try:
        VM_MARKER.parent.mkdir(parents=True, exist_ok=True)
        VM_MARKER.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    except OSError:
        pass
    if missing:
        log(f"[env] still missing after install: {missing} (run `sheng-chatterbox doctor`)")
    else:
        log("[env] all imports ok")
    return marker


def marker_state() -> dict[str, Any] | None:
    try:
        return json.loads(VM_MARKER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_installed() -> bool:
    """VM-local check: marker exists *and* the imports still work."""
    if marker_state() is None:
        return False
    probes = probe_imports()
    return all(probes.get(k) is not None for k in ("torch", "chatterbox", "transformers", "datasets"))


# ---------------------------------------------------------------- drive/mount
def mount_drive(log=print, *, mountpoint: str = "/content/drive") -> bool:
    """Mount Google Drive on Colab. Returns True when mounted (or not on Colab)."""
    if colab_drive_mounted():
        log("[env] Drive already mounted")
        return True
    if not Path("/content").is_dir():
        log("[env] not on Colab - using local artifacts dir")
        return False
    try:
        from google.colab import drive  # type: ignore

        log("[env] mounting Google Drive (approve the popup if it appears)")
        drive.mount(mountpoint, force_remount=False)
        return colab_drive_mounted()
    except Exception as exc:  # noqa: BLE001
        log(f"[env] Drive mount failed: {exc}")
        return False


# ----------------------------------------------------------------- model files
def download_pretrained(paths: Paths, *, log=print, force: bool = False, token: str | None = None) -> Path:
    """Snapshot the files the multilingual loader needs into ``paths.pretrained_dir``."""
    paths.pretrained_dir.mkdir(parents=True, exist_ok=True)
    present = all((paths.pretrained_dir / name).exists() for name in PRETRAINED_FILES)
    if present and not force:
        log(f"[model] already downloaded: {paths.pretrained_dir}")
        return paths.pretrained_dir

    from huggingface_hub import snapshot_download

    log(f"[model] downloading {MODEL_REPO_ID} -> {paths.pretrained_dir}")
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        allow_patterns=PRETRAINED_FILES,
        local_dir=str(paths.pretrained_dir),
        token=token or os.environ.get("HF_TOKEN"),
    )
    missing = [name for name in PRETRAINED_FILES if not (paths.pretrained_dir / name).exists()]
    if missing:
        raise RuntimeError(f"model download incomplete, missing: {missing}")
    log(f"[model] ready ({', '.join(PRETRAINED_FILES)})")
    return paths.pretrained_dir


# ---------------------------------------------------------------------- doctor
def doctor(paths: Paths) -> list[tuple[str, str, str]]:
    """Rows of (check, ok|warn|fail, detail) - rendered by the CLI."""
    rows: list[tuple[str, str, str]] = []

    gpu = gpu_info()
    if gpu.get("cuda"):
        detail = f"{gpu['name']}, {gpu['vram_gb']} GiB, cc {gpu.get('cc', '?')}, bf16={gpu['bf16']}"
        rows.append(("GPU", "ok", detail))
    elif "error" in gpu:
        rows.append(("GPU", "fail", gpu["error"]))
    else:
        rows.append(("GPU", "warn", "no CUDA device - only --dry-run works"))

    probes = probe_imports()
    for name, version in probes.items():
        if version is None:
            rows.append((f"import {name}", "fail", "missing - run `sheng-chatterbox env`"))
        else:
            rows.append((f"import {name}", "ok", str(version)))

    rows.append(("chatterbox version", "ok" if probes.get("chatterbox") else "warn",
                 f"expected {CHATTERBOX_VERSION}"))

    if on_drive(paths.artifacts):
        rows.append(("artifacts on Drive", "ok", str(paths.artifacts)))
    else:
        rows.append(
            ("artifacts on Drive", "warn", f"{paths.artifacts} - ephemeral on Colab, set SHENG_CHATTERBOX_ARTIFACTS")
        )

    free_gb = shutil.disk_usage(paths.artifacts if paths.artifacts.exists() else Path.cwd()).free / 2**30
    rows.append(("free disk", "ok" if free_gb > 8 else "warn", f"{free_gb:.1f} GiB"))

    for name, path in (
        ("train manifest", paths.train_manifest),
        ("val manifest", paths.val_manifest),
        ("base T3 weights", paths.base_t3_path),
    ):
        if path.exists():
            rows.append((name, "ok", str(path)))
        else:
            rows.append((name, "warn", f"missing: {path}"))

    if paths.train_manifest.exists():
        from . import manifest as M

        rep = M.validate_manifest(paths.train_manifest, sample_rate=16_000)
        rows.append(
            ("train manifest valid", "ok" if rep.ok else "fail",
             f"{rep.rows} clips, {rep.total_seconds / 60:.1f} min, {len(rep.errors)} errors")
        )
        if rep.speakers:
            rows.append(("speakers", "ok" if SPEAKER in rep.speakers else "warn", str(rep.speakers)))

    rows.append(("HF base T3 file", "ok", MTL_T3_MODEL))
    return rows


def tmp_dir(prefix: str = "sheng_chatterbox_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))
