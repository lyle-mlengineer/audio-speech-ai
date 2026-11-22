"""Stage 1: can this machine do the job?

Import-light on purpose. On a fresh Colab VM ``torch`` is present but
``f5_tts`` is not, and a readable answer beats an ImportError three stages
later.
"""

from __future__ import annotations

import importlib
import platform
import shutil
from pathlib import Path

from .config import DEFAULT_PROFILE, PROFILES, auto_profile

F5_PIN = "f5-tts==1.1.22"

INSTALL_HINT = f"""\
# on Colab (once per VM):
!git clone https://github.com/<you>/audio-speech-ai.git
%cd audio-speech-ai/finetune-sauti-tts
!pip install -q -e .
!pip install -q {F5_PIN}     # brings torch/torchaudio/vocos/datasets/ema-pytorch
!sauti-sheng setup
"""


def _version(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    return str(getattr(mod, "__version__", "installed"))


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _disk_gb(path: Path) -> float | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return round(shutil.disk_usage(path).free / 1024**3, 1)
    except Exception:
        return None


def report(artifacts: Path | None = None, scratch: Path | None = None) -> dict:
    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _version("torch"),
        "torchaudio": _version("torchaudio"),
        "f5_tts": _version("f5_tts"),
        "accelerate": _version("accelerate"),
        "datasets": _version("datasets"),
        "vocos": _version("vocos"),
        "soundfile": _version("soundfile"),
        "pyarrow": _version("pyarrow"),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "ffmpeg": _ffmpeg(),
        "cuda": False,
        "gpu": None,
        "gpu_gb": None,
        "bf16": False,
        "artifacts_free_gb": _disk_gb(artifacts) if artifacts else None,
        "scratch_free_gb": _disk_gb(scratch) if scratch else None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["cuda"] = True
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_gb"] = round(props.total_memory / 1024**3, 1)
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            try:
                info["bf16"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                info["bf16"] = False
    except Exception:
        pass
    return info


def check(info: dict | None = None) -> dict:
    info = info or report()
    problems: list[str] = []
    if info["torch"] is None:
        problems.append(f"torch is missing -> !pip install -q {F5_PIN}")
    if info["f5_tts"] is None:
        problems.append(f"f5-tts is missing -> !pip install -q {F5_PIN}")
    if info["soundfile"] is None:
        problems.append("soundfile is missing -> !pip install -q soundfile")
    if info["pyarrow"] is None:
        problems.append("pyarrow is missing -> !pip install -q pyarrow")
    if not info.get("cuda"):
        problems.append("no CUDA device: setup/data/model/export work, training does not")
    if info.get("gpu_gb") and info["gpu_gb"] < PROFILES[DEFAULT_PROFILE].min_vram_gb:
        problems.append(
            f"only {info['gpu_gb']} GB VRAM - F5 fine-tuning wants >= "
            f"{PROFILES[DEFAULT_PROFILE].min_vram_gb:.0f} GB (the {DEFAULT_PROFILE} profile)"
        )
    for key, label in (("artifacts_free_gb", "artifacts"), ("scratch_free_gb", "scratch")):
        free = info.get(key)
        if free is not None and free < 8:
            problems.append(f"{label} has {free} GB free; a run needs ~10 GB (checkpoints + raw model)")
    info["problems"] = problems
    info["suggested_profile"] = auto_profile(info.get("gpu"), info.get("gpu_gb"))
    if info["suggested_profile"] is None and info.get("cuda"):
        info["suggested_profile"] = DEFAULT_PROFILE
    return info


def run(artifacts: Path | None = None, scratch: Path | None = None) -> dict:
    info = check(report(artifacts, scratch))
    print("sauti-sheng setup")
    print(f"  python       {info['python']}  ({info['platform']})")
    print(f"  torch        {info['torch']}   torchaudio {info['torchaudio']}   cuda={info['cuda']}")
    print(f"  gpu          {info['gpu']} ({info['gpu_gb']} GB)   bf16={info['bf16']}")
    print(f"  f5-tts       {info['f5_tts']}   accelerate {info['accelerate']}   datasets {info['datasets']}")
    print(f"  vocos        {info['vocos']}   soundfile {info['soundfile']}   pyarrow {info['pyarrow']}")
    print(f"  ffmpeg       {info['ffmpeg']}")
    print(f"  disk free    artifacts {info['artifacts_free_gb']} GB   scratch {info['scratch_free_gb']} GB")
    print(f"  profile      {info['suggested_profile']}" + (
        "   (no bf16 on this GPU - the t4 profile trains in fp16)"
        if info.get("cuda") and not info.get("bf16") else ""
    ))
    if info["problems"]:
        print("\n  problems:")
        for problem in info["problems"]:
            print(f"   - {problem}")
        if info["f5_tts"] is None or info["torch"] is None:
            print("\n" + INSTALL_HINT)
    else:
        print("\n  all good.")
    return info


if __name__ == "__main__":
    run()
