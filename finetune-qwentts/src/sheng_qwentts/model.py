"""Model download / loading for Qwen3-TTS-12Hz-0.6B-Base.

Two things the upstream ``finetuning/comments`` do not tell you:

* the upstream trainer does ``shutil.copytree(MODEL_PATH, output_dir)`` for every
  checkpoint, so ``--init_model_path`` has to be a **local directory**, not a
  ``Qwen/...`` repo id;
* ``flash_attention_2`` needs a compiled flash-attn, which is a 10-20 min
  source build on Colab. ``sdpa`` (PyTorch native) is the default here; ask for
  flash-attn explicitly with ``--attn flash_attention_2`` (profile ``a100``).
"""

from __future__ import annotations

from pathlib import Path

from .paths import MODEL_REPO_ID, TOKENIZER_REPO_ID, Paths

DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def download_snapshots(paths: Paths, *, force: bool = False) -> dict[str, str]:
    """Copy both HF snapshots into the artifacts root (idempotent)."""
    from huggingface_hub import snapshot_download

    out: dict[str, str] = {}
    for repo, dest in ((MODEL_REPO_ID, paths.model_dir), (TOKENIZER_REPO_ID, paths.tokenizer_dir)):
        if dest.is_dir() and any(dest.glob("*.safetensors")) and not force:
            out[repo] = str(dest)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=str(dest))
        out[repo] = str(dest)
    return out


def require_local_model(paths: Paths) -> Path:
    model_dir = paths.model_dir
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"{model_dir} has no model.safetensors - run `sheng-qwentts model` first "
            "(the upstream trainer needs a local directory, not a repo id)"
        )
    return model_dir


def torch_dtype(name: str):
    import torch

    try:
        return getattr(torch, DTYPES[name])
    except KeyError as exc:
        raise ValueError(f"unknown dtype {name!r}; use bfloat16 / float16 / float32") from exc


def load_tts(model_path: str | Path, *, dtype: str = "bfloat16", attn: str = "sdpa", device: str = "cuda:0"):
    """``Qwen3TTSModel.from_pretrained`` with an sdpa fallback.

    Returns the wrapper (``.model`` is the HF model, ``.processor`` the text
    processor). Falls back to sdpa when the requested attention backend is not
    installed instead of dying 3 minutes into a Colab setup.
    """
    from qwen_tts import Qwen3TTSModel

    kwargs = dict(device_map=device, dtype=torch_dtype(dtype))
    try:
        return Qwen3TTSModel.from_pretrained(str(model_path), attn_implementation=attn, **kwargs)
    except (ImportError, ValueError, RuntimeError) as exc:
        if attn == "sdpa":
            raise
        print(f"[model] attention backend {attn!r} unavailable ({exc}); falling back to sdpa")
        return Qwen3TTSModel.from_pretrained(str(model_path), attn_implementation="sdpa", **kwargs)
