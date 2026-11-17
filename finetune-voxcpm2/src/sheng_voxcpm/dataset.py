"""Build the Sheng training data from the HF dataset.

Source: ``lyle-mlengineer/kenyan-celebs`` -> 42 clips, single speaker "abel",
30 s each, mono 22050 Hz in the parquet (the card claims 16 kHz; the files are
not, so we always resample ourselves).

Output (all under ``paths.artifacts``):

    data/wav16k/<speaker>_<idx:04d>.wav    16 kHz mono PCM16
    manifests/train.jsonl                  ~90% of clips
    manifests/val.jsonl                    held-out clips (real val, unseen text)
    manifests/dataset_stats.json           machine summary
    manifests/data_report.md               human summary used by progress.md

Clips are already 30 s, i.e. inside VoxCPM2's recommended 3-30 s window, so we
keep them whole - splitting would require forced alignment to keep the
transcripts word-accurate, and wrong transcripts are worse than long clips.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import manifest as M
from .paths import DATASET_ID, TARGET_SAMPLE_RATE, Paths

_WS = re.compile(r"\s+")
_ASCII_WORD = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")


@dataclass
class PrepareConfig:
    dataset_id: str = DATASET_ID
    speaker: str | None = None  # keep only this speaker (dataset has just "abel")
    target_sr: int = TARGET_SAMPLE_RATE
    val_clip_count: int = 4
    seed: int = 1337
    min_dur: float = M.MIN_CLIP_SEC
    max_dur: float = M.MAX_CLIP_SEC
    ref_audio_fraction: float = 0.0  # 0-1; see README "reference conditioning"
    force: bool = False


def clean_text(text: str) -> str:
    """Conservative normalisation: transcripts must stay word-for-word faithful."""
    text = (text or "").replace("\u00a0", " ").replace("\r", " ").replace("\n", " ")
    text = _WS.sub(" ", text).strip()
    return text


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", (name or "speaker")).strip("-").lower() or "speaker"


def _resample(array, src_sr: int, dst_sr: int):
    if src_sr == dst_sr:
        return array
    try:
        import torch
        import torchaudio

        wav = torch.as_tensor(array, dtype=torch.float32)
        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        return torchaudio.functional.resample(wav, src_sr, dst_sr).numpy()
    except Exception:  # noqa: BLE001 - torchaudio/resample kernel may be missing
        import numpy as np
        from scipy.signal import resample_poly

        arr = np.asarray(array, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        g = np.gcd(int(src_sr), int(dst_sr))
        return resample_poly(arr, int(dst_sr) // g, int(src_sr) // g).astype("float32")


def _write_wav(path: Path, array, sample_rate: int) -> float:
    import numpy as np
    import soundfile as sf

    arr = np.asarray(array, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:  # integer samples decoded as float
        arr = arr / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sample_rate, subtype="PCM_16")
    return float(len(arr)) / sample_rate


def code_switch_stats(texts: list[str]) -> dict[str, float]:
    """Rough English-leaning token share - a sanity signal that we have Sheng."""
    ascii_words = total = 0
    for text in texts:
        for tok in text.split():
            total += 1
            if _ASCII_WORD.match(tok.strip(".,!?;:\"'()")):
                ascii_words += 1
    return {
        "tokens": total,
        "ascii_word_share": round(ascii_words / total, 3) if total else 0.0,
    }


def is_prepared(paths: Paths, cfg: PrepareConfig) -> bool:
    if not (paths.train_manifest.exists() and paths.val_manifest.exists() and paths.data_stats.exists()):
        return False
    try:
        stats = json.loads(paths.data_stats.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        stats.get("dataset_id") == cfg.dataset_id
        and stats.get("target_sr") == cfg.target_sr
        and stats.get("seed") == cfg.seed
        and stats.get("val_clip_count") == cfg.val_clip_count
        and stats.get("ref_audio_fraction") == cfg.ref_audio_fraction
        and stats.get("speaker_filter") == cfg.speaker
    )


def prepare(paths: Paths, cfg: PrepareConfig | None = None, *, log=print) -> dict[str, Any]:
    """Download -> 16 kHz WAVs -> train/val manifests. Idempotent."""
    cfg = cfg or PrepareConfig()
    paths.ensure()

    if is_prepared(paths, cfg) and not cfg.force:
        log(f"[data] already prepared, reusing {paths.manifest_dir}")
        return json.loads(paths.data_stats.read_text(encoding="utf-8"))

    from datasets import load_dataset

    log(f"[data] loading HF dataset {cfg.dataset_id}")
    ds = load_dataset(cfg.dataset_id, split="train")
    log(f"[data] {len(ds)} rows, columns={ds.column_names}")

    clips: list[dict[str, Any]] = []
    dropped: list[str] = []
    for i in range(len(ds)):
        row = ds[i]
        speaker = str(row.get("speaker") or "speaker")
        if cfg.speaker and speaker != cfg.speaker:
            dropped.append(f"row {i}: speaker {speaker!r} filtered out")
            continue
        text = clean_text(row.get("text") or "")
        if not text:
            dropped.append(f"row {i}: empty transcript")
            continue

        audio = row["audio"]
        array, src_sr = audio["array"], int(audio["sampling_rate"])
        arr16 = _resample(array, src_sr, cfg.target_sr)
        name = f"{_slug(speaker)}_{i:04d}.wav"
        out = paths.wav_dir / name
        dur = _write_wav(out, arr16, cfg.target_sr)
        if dur < cfg.min_dur:
            out.unlink(missing_ok=True)
            dropped.append(f"row {i}: {dur:.2f}s shorter than {cfg.min_dur}s")
            continue
        if dur > cfg.max_dur:
            dropped.append(f"row {i}: {dur:.2f}s longer than {cfg.max_dur}s (kept, watch VRAM)")

        clips.append(
            {
                "audio": paths.rel_to_manifest(out),
                "text": text,
                "duration": round(dur, 3),
                "dataset_id": 0,
                "speaker": speaker,
                "_src_sr": src_sr,
            }
        )

    if not clips:
        raise RuntimeError("no usable clips after filtering - refusing to write empty manifests")

    # deterministic split; the val clips are real held-out speech (unseen text)
    order = list(range(len(clips)))
    random.Random(cfg.seed).shuffle(order)
    n_val = max(1, min(cfg.val_clip_count, len(clips) // 5)) if len(clips) >= 5 else 0
    val_idx = set(order[:n_val])
    train_rows = [c for i, c in enumerate(clips) if i not in val_idx]
    val_rows = [c for i, c in enumerate(clips) if i in val_idx]

    # optional reference-audio conditioning (official rec: 30-50% of samples)
    if cfg.ref_audio_fraction > 0 and len(train_rows) > 2:
        rng = random.Random(cfg.seed + 1)
        n_ref = max(1, int(round(cfg.ref_audio_fraction * len(train_rows))))
        targets = rng.sample(range(len(train_rows)), n_ref)
        for t in targets:
            pool = [r["audio"] for j, r in enumerate(train_rows) if j != t]
            train_rows[t]["ref_audio"] = rng.choice(pool)

    n_train = M.write_manifest(paths.train_manifest, train_rows)
    n_val = M.write_manifest(paths.val_manifest, val_rows)
    log(f"[data] wrote {n_train} train / {n_val} val clips")

    train_rep = M.validate_manifest(
        paths.train_manifest, sample_rate=cfg.target_sr, min_sec=cfg.min_dur, max_sec=cfg.max_dur
    )
    val_rep = M.validate_manifest(
        paths.val_manifest, sample_rate=cfg.target_sr, min_sec=cfg.min_dur, max_sec=cfg.max_dur
    )

    stats = {
        "dataset_id": cfg.dataset_id,
        "target_sr": cfg.target_sr,
        "seed": cfg.seed,
        "val_clip_count": cfg.val_clip_count,
        "ref_audio_fraction": cfg.ref_audio_fraction,
        "speaker_filter": cfg.speaker,
        "drops": dropped,
        "train": train_rep.summary(),
        "val": val_rep.summary(),
        "code_switch": code_switch_stats([r["text"] for r in train_rows]),
        "source_sample_rates": sorted({c["_src_sr"] for c in clips}),
    }
    paths.data_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    M.write_data_report(
        paths.data_report,
        [train_rep, val_rep],
        extra={
            "dataset": cfg.dataset_id,
            "dataset type": "Sheng (Swahili + English code-switching), single speaker",
            "target sample rate": f"{cfg.target_sr} Hz (AudioVAE encoder input)",
            "seed": cfg.seed,
            "dropped/notes": len(dropped),
        },
    )

    if not train_rep.ok or not val_rep.ok:
        raise RuntimeError(f"manifest validation failed, see {paths.data_report}")

    log(f"[data] train {train_rep.total_seconds / 60:.1f} min | val {val_rep.total_seconds / 60:.1f} min")
    return stats
