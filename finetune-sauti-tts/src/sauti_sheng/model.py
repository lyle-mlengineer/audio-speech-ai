"""Stage 3: turn MsingiAI's 5.4 GB release into something F5 can fine-tune from.

What the release actually is
----------------------------
``msingiai/sauti-tts`` ships ``model_last.pt``, a **full training checkpoint**:

.. code-block:: text

    model_state_dict        online weights
    ema_model_state_dict    EMA weights, keys prefixed ``ema_model.``
    optimizer_state_dict    AdamW moments            (~4 GB)
    scheduler_state_dict    LinearLR chain
    update                  15350

F5's ``Trainer.load_checkpoint`` decides between "resume this run" and
"initialise from these weights" by looking for ``update``/``step`` - not by
filename. Hand it the release file as ``pretrained_sauti.pt`` and it resumes:
optimizer restored, LR schedule already at its floor, step counter at 15350.
That is not a fine-tune, and it is silent.

So this stage extracts ``{"ema_model_state_dict": ...}`` into
``pretrained/sauti-tts/sauti_ema.pt`` (~1.35 GB, Drive-safe), verifies the text
embedding matches Sauti's ``vocab.txt`` (2545 entries -> 2546 embedding rows,
because F5 pads the text embedding with one filler row), writes a report, and
deletes the 5.4 GB raw file from scratch.

The EMA is the right starting point for a *fine-tune* (it is the smoothed
version of a finished model). For evaluating that fine-tune, the online weights
are the right ones - see ``checkpoint.select_state_dict``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .checkpoint import (
    EMA_KEY,
    describe,
    prepare_for_load,
    select_state_dict,
    strip_to_ema,
    text_embed_rows,
)
from .compat import free_memory
from .paths import MODEL_FILE, MODEL_REPO_ID, SAMPLE_RATE, Paths
from .text import read_vocab

#: ``F5TTS_v1_Base`` - Sauti is a v1 Base fine-tune. The v1 config is the one
#: without ``pe_attn_head``/``text_mask_padding`` (those are the older v0 model).
DIT_CONFIG: dict = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
)

#: Shared by training and inference; must match how Sauti was trained.
MEL_KWARGS: dict = dict(
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=100,
    target_sample_rate=SAMPLE_RATE,
    mel_spec_type="vocos",
)

HOP_LENGTH = MEL_KWARGS["hop_length"]


# ------------------------------------------------------------------ downloads
def download_raw(paths: Paths, force: bool = False) -> Path:
    """Fetch ``model_last.pt`` into scratch (never Drive if scratch exists)."""
    target = paths.sauti_raw_ckpt
    if target.is_file() and not force:
        return target
    from huggingface_hub import hf_hub_download

    target.parent.mkdir(parents=True, exist_ok=True)
    got = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILE,
            local_dir=str(target.parent),
        )
    )
    if got.resolve() != target.resolve():
        shutil.move(str(got), target)
    return target


def _load_raw(path: Path) -> dict:
    """Load the 5.4 GB checkpoint with as little RAM as torch allows."""
    import torch

    attempts = (
        dict(map_location="cpu", weights_only=True, mmap=True),
        dict(map_location="cpu", weights_only=True),
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return torch.load(str(path), **kwargs)
        except TypeError as exc:  # torch too old for ``mmap``
            last_error = exc
        except RuntimeError as exc:
            # "mmap can only be used with files saved with torch.save(...)"
            last_error = exc
    raise RuntimeError(f"could not load {path}: {last_error}")


def _delete_raw(paths: Paths) -> bool:
    """Only delete a file we downloaded ourselves into scratch."""
    if paths.scratch is None:
        return False
    raw = paths.sauti_raw_ckpt
    try:
        raw.relative_to(paths.scratch)
    except ValueError:
        return False
    if raw.is_file():
        raw.unlink()
        return True
    return False


# ------------------------------------------------------------------- report
def _write_report(paths: Paths, raw: Path, raw_bytes: int, info: dict, vocab_size: int, deleted: bool) -> None:
    ema_gb = info["ema_bytes"] / 1024**3
    lines = [
        "# Model report - msingiai/sauti-tts",
        "",
        f"- source: `{MODEL_REPO_ID}/{MODEL_FILE}`",
        f"- downloaded: {raw_bytes / 1024**3:.2f} GB -> `{paths.rel(raw)}`",
        f"- top-level keys: `{', '.join(info['keys'])}`",
        f"- training checkpoint: **{info['training_checkpoint']}**"
        + (f" (update {info['update']})" if info["update"] is not None else ""),
        f"- EMA tensors: {info['ema_tensors']} ({ema_gb:.2f} GB)",
        f"- text embedding rows: **{info['text_embed_rows']}** (= {vocab_size} chars + 1 filler row)",
        f"- vocab file entries: **{vocab_size}**",
        "",
        "## What was kept",
        "",
        (
            f"`{paths.rel(paths.sauti_ema_ckpt)}` contains exactly one key,"
            f" `{EMA_KEY}`, with the ``ema_model.`` prefix intact (F5 strips it)."
        ),
        f"Size: **{ema_gb:.2f} GB**.",
        "",
        "## What was dropped, and why",
        "",
        "| key | why |",
        "|---|---|",
        "| `optimizer_state_dict` | ~4 GB of AdamW moments for a run we are not resuming |",
        "| `scheduler_state_dict` | belongs to update 15350's finished schedule |",
        f"| `update` ({info['update']}) | its presence makes F5 treat the file as a **resume**, not a pretrained init |",
        "| `model_state_dict` | the online weights of a finished run; EMA is the better init |",
        "",
        (
            "> F5 chooses the branch by checkpoint keys, not by filename. A file named"
            " `pretrained_*.pt` that still contains `update` is still resumed. That is"
            " why the strip is done here instead of a rename."
        ),
        "",
        "## Vocab check",
        "",
        (
            "Sauti's `vocab.txt` has one entry per line, index 0 = space (F5's"
            " unknown-character fallback). F5's `TextEmbedding` allocates"
            " `nn.Embedding(text_num_embeds + 1, text_dim)` for its filler token, so"
            " `transformer.text_embed.text_embed.weight` has"
        ),
        (
            f" **{vocab_size} + 1 = {info['text_embed_rows']} rows** - one more than the"
            " vocab file has lines, and no character is remapped. Training must reuse"
            " **Sauti's** vocab: F5's Emilia vocab would silently remap every character."
        ),
        "",
    ]
    if deleted:
        lines += [
            "## Cleanup",
            "",
            (
                f"The {raw_bytes / 1024**3:.2f} GB raw download was deleted from scratch;"
                " only the EMA file is kept."
            ),
            "",
        ]
    paths.model_report.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------- stage
def extract(paths: Paths, force: bool = False) -> dict:
    """Download -> inspect -> strip to EMA -> verify -> delete raw."""
    import torch

    paths.ensure()
    if paths.sauti_ema_ckpt.is_file() and paths.model_report.is_file() and not force:
        return _cached_summary(paths)

    from .data import ensure_vocab

    ensure_vocab(paths)
    vocab_size = len(read_vocab(paths.sauti_vocab))
    raw = download_raw(paths, force=force)
    raw_bytes = raw.stat().st_size

    checkpoint = _load_raw(raw)
    info = describe(checkpoint)
    expected_rows = text_embed_rows(vocab_size)
    if info["text_embed_rows"] not in (None, expected_rows):
        raise RuntimeError(
            f"checkpoint text embedding has {info['text_embed_rows']} rows but "
            f"{paths.sauti_vocab} has {vocab_size} entries, so F5 would build "
            f"{expected_rows} rows; wrong vocab file - training would silently remap "
            "every character"
        )
    if not info["training_checkpoint"]:
        raise RuntimeError(
            f"{raw} has no update/optimizer keys; it does not look like the Sauti release"
        )

    ema = strip_to_ema(checkpoint)
    tmp = paths.sauti_ema_ckpt.with_suffix(".pt.tmp")
    torch.save(ema, tmp)
    tmp.replace(paths.sauti_ema_ckpt)

    del ema
    del checkpoint
    free_memory()
    deleted = _delete_raw(paths)

    _write_report(paths, raw, raw_bytes, info, vocab_size, deleted)
    summary = {
        "raw_bytes": raw_bytes,
        "ema_bytes": paths.sauti_ema_ckpt.stat().st_size,
        "ema_tensors": info["ema_tensors"],
        "text_embed_rows": info["text_embed_rows"],
        "text_vocab_size": vocab_size,
        "vocab_size": vocab_size,
        "update": info["update"],
        "raw_deleted": deleted,
        "checkpoint": str(paths.sauti_ema_ckpt),
        "report": str(paths.model_report),
    }
    (paths.pretrained_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _cached_summary(paths: Paths) -> dict:
    cached = paths.pretrained_dir / "model_summary.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    return {
        "ema_bytes": paths.sauti_ema_ckpt.stat().st_size,
        "checkpoint": str(paths.sauti_ema_ckpt),
        "report": str(paths.model_report),
        "cached": True,
    }


# ------------------------------------------------------------------- loading
def build_cfm(vocab_path: Path):
    """Construct an F5 ``CFM`` with the v1 Base DiT and Sauti's vocab."""
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    transformer = DiT(**DIT_CONFIG, text_num_embeds=vocab_size, mel_dim=MEL_KWARGS["n_mel_channels"])
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=dict(MEL_KWARGS),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    )
    return model, vocab_size


def load_weights(model, checkpoint_path: Path, use_ema: bool = False):
    """Load a checkpoint (raw or F5 training file) into ``model``."""
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = prepare_for_load(select_state_dict(checkpoint, use_ema=use_ema))
    model.load_state_dict(state, strict=True)
    del checkpoint
    free_memory()
    return model


def load_for_inference(
    paths: Paths,
    checkpoint: Path | str | None = None,
    *,
    use_ema: bool = False,
    device: str = "cuda",
    vocab: Path | None = None,
):
    """Build + load a Sauti/F5 model ready for ``.sample()``.

    Defaults to the EMA weights, i.e. the untouched Sauti release, when no
    checkpoint is given. Pass a fine-tuned checkpoint with ``use_ema=False``.
    """
    vocab = Path(vocab or paths.sauti_vocab)
    checkpoint = Path(checkpoint or paths.sauti_ema_ckpt)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} missing - run `sauti-sheng model` first")
    model, vocab_size = build_cfm(vocab)
    model = load_weights(model, checkpoint, use_ema=use_ema)
    return model.to(device).eval(), vocab_size
