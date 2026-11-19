"""Stage 1: is this machine able to do the job?

Deliberately import-light. On a fresh Colab VM ``torch`` is there but
``qwen-tts`` is not, and you want a readable answer instead of an ImportError
traceback three stages later.
"""

from __future__ import annotations

import importlib
import platform

PINNED_TRANSFORMERS = "4.57.3"

INSTALL_HINT = """\
# on Colab (run once per VM):
!git clone https://github.com/<you>/audio-speech-ai.git
%cd audio-speech-ai/finetune-qwentts
!pip install -q -e .
!pip install -q -U qwen-tts      # pulls transformers==4.57.3, librosa, soundfile
# optional, only for --profile a100 (10-20 min source build):
# !pip install -q -U flash-attn --no-build-isolation
"""


def _version(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    return str(getattr(mod, "__version__", "installed"))


def report() -> dict:
    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "qwen_tts": _version("qwen_tts"),
        "soundfile": _version("soundfile"),
        "pyarrow": _version("pyarrow"),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "flash_attn": _version("flash_attn"),
        "cuda": False,
        "gpu": None,
        "gpu_gb": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda"] = True
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            info["device_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception:
        pass
    return info


def suggest_profile(info: dict) -> str:
    """Map the GPU to a training profile (see config.PROFILES)."""
    name = (info.get("gpu") or "").upper()
    gb = info.get("gpu_gb") or 0
    if not info.get("cuda"):
        return "cpu"
    if "A100" in name or gb >= 38:
        return "a100"
    if "L4" in name or gb >= 22:
        return "l4"
    if gb >= 14:
        return "t4"
    return "cpu"


def check(info: dict | None = None) -> dict:
    """Return the report plus a list of problems that would break the run."""
    info = info or report()
    problems: list[str] = []
    if info["torch"] is None:
        problems.append("torch is missing -> !pip install -q -U qwen-tts")
    if info["qwen_tts"] is None:
        problems.append("qwen-tts is missing -> !pip install -q -U qwen-tts")
    elif info["transformers"] and info["transformers"] != PINNED_TRANSFORMERS:
        problems.append(
            f"transformers {info['transformers']} != {PINNED_TRANSFORMERS} (the version qwen-tts pins); "
            "mismatched versions are the #1 source of 'AutoModel returned <other class>' errors"
        )
    if info["soundfile"] is None:
        problems.append("soundfile is missing -> !pip install -q soundfile")
    if info["pyarrow"] is None:
        problems.append("pyarrow is missing -> !pip install -q pyarrow")
    if not info.get("cuda"):
        problems.append("no CUDA device: setup/data/model work, training does not")
    info["problems"] = problems
    info["suggested_profile"] = suggest_profile(info)
    return info


def run() -> dict:
    info = check()
    print("sheng-qwentts setup")
    print(f"  python      {info['python']}  ({info['platform']})")
    print(f"  torch       {info['torch']}   cuda={info['cuda']}")
    print(f"  gpu         {info['gpu']} ({info['gpu_gb']} GB)")
    print(f"  transformers {info['transformers']}  qwen-tts {info['qwen_tts']}")
    print(f"  flash-attn  {info['flash_attn']}")
    print(f"  profile     {info['suggested_profile']}")
    if info["problems"]:
        print("\n  problems:")
        for problem in info["problems"]:
            print(f"   - {problem}")
        if info["qwen_tts"] is None:
            print("\n" + INSTALL_HINT)
    else:
        print("\n  all good.")
    return info


if __name__ == "__main__":
    run()
