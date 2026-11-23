"""Pure helpers for reading/stripping F5-TTS checkpoints.

No torch import: tests use duck-typed fake tensors. ``model.py`` does the real
loading.

Why this module exists
----------------------
``msingiai/sauti-tts/model_last.pt`` is a *training* checkpoint. It contains
``model_state_dict``, ``ema_model_state_dict``, ``optimizer_state_dict``,
``scheduler_state_dict`` **and ``update``**. F5-TTS's ``Trainer.load_checkpoint``
treats the presence of ``update`` as "resume this run": it restores the
optimizer, restores the scheduler and starts counting at update 15350. That is
the opposite of fine-tuning - the LR schedule is already at its floor.

Renaming the file to ``pretrained_*.pt`` is **not enough**; the branch is chosen
by checkpoint keys, not by filename. The fix is to strip it down to a single
``{"ema_model_state_dict": ...}`` key. Then F5 takes the "initialise from
pretrained" branch and starts at update 0 with a fresh optimizer and schedule.
"""

from __future__ import annotations

EMA_KEY = "ema_model_state_dict"
ONLINE_KEY = "model_state_dict"
#: Keys that make F5 treat the file as a resumable run rather than a pretrained
#: init. ``update`` is the one Sauti actually has.
RESUME_KEYS = ("update", "step", "optimizer_state_dict", "scheduler_state_dict")


def top_level_keys(checkpoint: dict) -> list[str]:
    return sorted(str(k) for k in checkpoint)


def has_ema(checkpoint: dict) -> bool:
    return isinstance(checkpoint.get(EMA_KEY), dict) and bool(checkpoint[EMA_KEY])


def looks_like_training_checkpoint(checkpoint: dict) -> bool:
    return any(k in checkpoint for k in RESUME_KEYS)


def strip_to_ema(checkpoint: dict) -> dict:
    """Return an EMA-only checkpoint that F5 accepts as a pretrained init."""
    if not has_ema(checkpoint):
        raise KeyError(
            f"checkpoint has no {EMA_KEY!r}; found {top_level_keys(checkpoint)}. "
            "Sauti's release should have one - check you downloaded model_last.pt."
        )
    return {EMA_KEY: dict(checkpoint[EMA_KEY])}


#: Buffers that older F5-TTS releases saved and current ones compute. Loading a
#: checkpoint that contains them into a current model fails with "unexpected
#: keys", so both F5 and this module drop them.
STALE_KEYS = (
    "mel_spec.mel_stft.mel_scale.fb",
    "mel_spec.mel_stft.spectrogram.window",
)


def _unprefix(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        name = str(key).removeprefix("ema_model.")
        if name in {"initted", "step", "update"}:
            continue
        out[name] = value
    return out


def select_state_dict(checkpoint: dict, use_ema: bool = False) -> dict:
    """Pick the weights to load, without the ``ema_model.`` prefix.

    ``use_ema=False`` is the right choice for a *fine-tuned* checkpoint: F5's
    own docs warn that the EMA lags behind when you start from a finished model
    and only train for a few hundred updates. The untouched Sauti release is the
    opposite case - its online weights are from the last training step and its
    EMA is the smoothed, better one - so base-model inference uses EMA.
    """
    ema = checkpoint.get(EMA_KEY) if isinstance(checkpoint.get(EMA_KEY), dict) else {}
    online = checkpoint.get(ONLINE_KEY) if isinstance(checkpoint.get(ONLINE_KEY), dict) else {}

    if use_ema:
        if ema:
            return _unprefix(ema)
        if online:
            return dict(online)
    else:
        if online:
            return dict(online)
        if ema:
            return _unprefix(ema)
    raise KeyError(
        f"checkpoint has neither {ONLINE_KEY!r} nor {EMA_KEY!r}; found {top_level_keys(checkpoint)}"
    )


def prepare_for_load(state: dict) -> dict:
    """Drop the stale mel buffers F5 no longer stores."""
    return {k: v for k, v in state.items() if k not in STALE_KEYS}


def export_state(checkpoint: dict, use_ema: bool = False) -> dict:
    """Single-key checkpoint that F5's ``load_checkpoint`` understands.

    ``use_ema=True`` keeps the ``ema_model.`` key prefix, because F5 strips it
    itself; ``use_ema=False`` writes plain online weights.
    """
    if use_ema:
        if not has_ema(checkpoint):
            raise KeyError(f"cannot export EMA weights: no {EMA_KEY!r}")
        return {EMA_KEY: dict(checkpoint[EMA_KEY])}
    return {ONLINE_KEY: select_state_dict(checkpoint, use_ema=False)}


def strip_to_online(checkpoint: dict) -> dict:
    """Return an online-weights-only checkpoint (``use_ema=False`` inference)."""
    return {ONLINE_KEY: select_state_dict(checkpoint, use_ema=False)}


#: F5's ``TextEmbedding`` does ``nn.Embedding(text_num_embeds + 1, text_dim)``
#: ("use 0 as filler token"), so the checkpoint's text embedding always has one
#: row more than the vocab file has lines. ``F5TTS_v1_Base`` is the proof: its
#: ``vocab.txt`` has 2545 lines and its ``text_embed.weight`` is ``[2546, 512]``.
TEXT_EMBED_EXTRA_ROWS = 1


def text_embed_rows(vocab_size: int) -> int:
    """Rows F5's text embedding has for a vocab of ``vocab_size`` characters."""
    return int(vocab_size) + TEXT_EMBED_EXTRA_ROWS


def vocab_size_from_rows(rows: int) -> int:
    """Tokenizer size implied by a checkpoint's text-embedding row count."""
    return int(rows) - TEXT_EMBED_EXTRA_ROWS


def text_vocab_size(state: dict) -> int | None:
    """Rows of the text embedding.

    This is *not* the vocab size: F5 adds a filler row, so subtract
    :data:`TEXT_EMBED_EXTRA_ROWS` (or use :func:`vocab_size_from_rows`) before
    comparing with ``len(read_vocab(...))``.
    """
    for key, value in state.items():
        if str(key).endswith("transformer.text_embed.text_embed.weight"):
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) >= 1:
                return int(shape[0])
    return None


def state_nbytes(state: dict) -> int:
    total = 0
    for value in state.values():
        try:
            total += int(value.numel()) * int(value.element_size())
        except AttributeError:
            continue
    return total


def state_tensor_count(state: dict) -> int:
    return sum(1 for v in state.values() if hasattr(v, "shape"))


def describe(checkpoint: dict) -> dict:
    """Small JSON-able summary for the model report."""
    ema = checkpoint.get(EMA_KEY) if isinstance(checkpoint.get(EMA_KEY), dict) else {}
    online = checkpoint.get(ONLINE_KEY) if isinstance(checkpoint.get(ONLINE_KEY), dict) else {}
    rows = text_vocab_size(ema) or text_vocab_size(online)
    return {
        "keys": top_level_keys(checkpoint),
        "training_checkpoint": looks_like_training_checkpoint(checkpoint),
        "ema_tensors": state_tensor_count(ema),
        "ema_bytes": state_nbytes(ema),
        "online_tensors": state_tensor_count(online),
        "online_bytes": state_nbytes(online),
        "text_embed_rows": rows,
        "text_vocab_size": vocab_size_from_rows(rows) if rows is not None else None,
        "update": checkpoint.get("update"),
    }
