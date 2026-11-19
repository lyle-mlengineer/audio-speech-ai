"""Stage 2: the dataset -> 24 kHz WAVs + the JSONL the Qwen3-TTS trainer eats.

The Hugging Face dataset ``lyle-mlengineer/kenyan-celebs`` is one speaker
(``abel``) speaking Sheng: 42 clips of exactly 30.0 s, mono 16-bit WAV at
22.05 kHz, with transcript. Two things about it matter for this model:

* Qwen3-TTS only accepts **24 kHz** audio (the tokenizer, the ECAPA speaker
  encoder and the training mel all assert it), so everything is resampled once
  here;
* upstream single-speaker fine-tuning wants every line to point at the **same**
  ``ref_audio`` - that is the clip whose speaker embedding is baked into the
  trained checkpoint. We pick one training clip and use it everywhere.

No ``datasets`` dependency: the parquet is read with pyarrow and decoded with
soundfile, so this stage also runs on a bare machine.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .paths import (
    DATASET_ID,
    DATASET_PARQUET,
    SAMPLE_RATE,
    SOURCE_SAMPLE_RATE,
    SPEAKER,
    Paths,
)
from .resample import resample, to_mono

MIN_SECONDS = 1.0


@dataclass
class Clip:
    index: int
    text: str
    speaker: str
    seconds: float
    wav: Path


def download_parquet(paths: Paths, *, force: bool = False) -> Path:
    """Fetch the single parquet file (52 MB) into the artifacts root."""
    dest = paths.raw_parquet
    if dest.is_file() and not force and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        got = hf_hub_download(
            repo_id=DATASET_ID,
            filename=DATASET_PARQUET,
            repo_type="dataset",
            local_dir=str(paths.data_dir),
        )
        src = Path(got)
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
    except Exception:  # no network cache / no huggingface_hub: plain HTTP
        from urllib.request import urlopen

        url = (
            f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/{DATASET_PARQUET}"
        )
        with urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())
    return dest


def read_clips(parquet: Path, wav_dir: Path, *, force: bool = False) -> list[Clip]:
    """Decode every row, resample to 24 kHz, write one WAV per row."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet)
    rows = table.to_pylist()
    wav_dir.mkdir(parents=True, exist_ok=True)

    clips: list[Clip] = []
    for i, row in enumerate(rows, start=1):
        wav_path = wav_dir / f"utt_{i:04d}.wav"
        info = sf.info(io.BytesIO(row["audio"]["bytes"]))
        if wav_path.is_file() and not force:
            clips.append(Clip(i, row["text"], row.get("speaker", SPEAKER), float(info.duration), wav_path))
            continue
        wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
        wav = to_mono(wav)
        src_sr = int(sr)
        if src_sr != SAMPLE_RATE:
            wav = resample(wav, src_sr, SAMPLE_RATE)
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
        sf.write(wav_path, wav, SAMPLE_RATE, subtype="PCM_16")
        clips.append(
            Clip(
                index=i,
                text=str(row["text"]).strip(),
                speaker=str(row.get("speaker", SPEAKER)),
                seconds=wav.size / float(SAMPLE_RATE),
                wav=wav_path,
            )
        )
    return clips


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(
    paths: Paths,
    *,
    val_size: int = 4,
    ref_index: int | None = None,
    seed: int = 1337,
    force: bool = False,
) -> dict:
    """Full data stage. Returns a summary dict (also written to dataset_info.json)."""
    paths.ensure()
    if not val_size >= 0:
        raise ValueError("val_size must be >= 0")

    parquet = download_parquet(paths, force=force)
    clips = read_clips(parquet, paths.wav_dir, force=force)
    if len(clips) < 10:
        raise RuntimeError(f"only {len(clips)} clips in {DATASET_ID}; that is not enough to fine-tune")
    short = [c for c in clips if c.seconds < MIN_SECONDS]
    if short:
        raise RuntimeError(f"{len(short)} clip(s) shorter than {MIN_SECONDS}s: {[c.wav.name for c in short]}")

    order = list(range(len(clips)))
    random.Random(seed).shuffle(order)
    val_ids = set(order[:val_size])
    train = [c for c in clips if (c.index - 1) not in val_ids]
    val = [c for c in clips if (c.index - 1) in val_ids]
    if not train:
        raise RuntimeError("no training clips left after the validation split")

    # Reference speaker clip: the longest training clip unless one is forced.
    # It is copied to data/ref_audio.wav so a manual `python prepare_data.py` run
    # and the CLI both point at the same file.
    if ref_index is None:
        ref = max(train, key=lambda c: (c.seconds, -c.index))
    else:
        matches = [c for c in train if c.index == ref_index]
        if not matches:
            raise ValueError(f"--ref-index {ref_index} is not a training clip (1..{len(clips)})")
        ref = matches[0]
    ref_wav = paths.data_dir / "ref_audio.wav"
    ref_wav.write_bytes(ref.wav.read_bytes())

    train_rows = [
        {"audio": str(c.wav), "text": c.text, "ref_audio": str(ref_wav), "duration": round(c.seconds, 3)}
        for c in train
    ]
    val_rows = [
        {"audio": str(c.wav), "text": c.text, "ref_audio": str(ref_wav), "duration": round(c.seconds, 3)}
        for c in val
    ]
    _write_jsonl(paths.train_raw_jsonl, train_rows)
    if val_rows:
        _write_jsonl(paths.val_raw_jsonl, val_rows)

    total = sum(c.seconds for c in clips)
    info = {
        "dataset": DATASET_ID,
        "speaker": SPEAKER,
        "clips": len(clips),
        "sample_rate": SAMPLE_RATE,
        "source_sample_rate": SOURCE_SAMPLE_RATE,
        "total_seconds": round(total, 2),
        "total_minutes": round(total / 60.0, 2),
        "clip_seconds_min": round(min(c.seconds for c in clips), 2),
        "clip_seconds_max": round(max(c.seconds for c in clips), 2),
        "clip_seconds_mean": round(total / len(clips), 2),
        "train_clips": len(train_rows),
        "train_seconds": round(sum(c.seconds for c in train), 2),
        "val_clips": len(val_rows),
        "ref_clip": ref.wav.name,
        "ref_seconds": round(ref.seconds, 2),
        "ref_audio": str(ref_wav),
        "seed": seed,
        "val_indices": sorted(c.index for c in val),
        "mean_chars": round(sum(len(c.text) for c in clips) / len(clips), 1),
    }
    paths.dataset_info.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    paths.data_report.write_text(render_report(info, clips, val), encoding="utf-8", newline="\n")

    _write_ref_text(paths, ref)
    return info


def _write_ref_text(paths: Paths, ref: Clip) -> None:
    """Keep the reference transcript next to the reference audio (docs + eval)."""
    (paths.data_dir / "ref_text.txt").write_text(ref.text + "\n", encoding="utf-8", newline="\n")


def render_report(info: dict, clips: list[Clip], val: list[Clip]) -> str:
    chars_per_sec = info["mean_chars"] / info["clip_seconds_mean"]
    lines = [
        f"# Data report - {info['dataset']}",
        "",
        f"* speaker: `{info['speaker']}` (single speaker, Sheng / Swahili + English code switching)",
        f"* clips: **{info['clips']}** at {info['clip_seconds_mean']:.1f} s "
        f"({info['total_minutes']:.1f} min total)",
        f"* written as {info['sample_rate']} Hz mono PCM16 (source was {info['source_sample_rate']} Hz)",
        f"* split: {info['train_clips']} train / {info['val_clips']} val (seed {info['seed']})",
        f"* reference clip for the speaker embedding: `{info['ref_clip']}` ({info['ref_seconds']:.1f} s)",
        "",
        "## What this means for fine-tuning",
        "",
        f"* ~{chars_per_sec:.0f} characters per second of Sheng; ~{info['total_seconds'] / 60:.1f} min of audio "
        "is a *small* single-speaker fine-tune - expect the voice to transfer, "
        "not a new vocabulary. Qwen3-TTS already speaks Swahili and English, so "
        "this is mostly teaching it the speaker + the Sheng delivery.",
        "* every clip is exactly the same length, i.e. the data was cut on a grid "
        "from one recording; some cuts land mid-word. The transcripts are "
        "verbatim, so leave them alone (rewriting them is what ruins alignment).",
        "",
        "| # | wav | seconds | chars | split |",
        "|---|---|---|---|---|",
    ]
    val_ids = {c.index for c in val}
    for c in clips:
        split = "val" if c.index in val_ids else "train"
        lines.append(f"| {c.index} | `{c.wav.name}` | {c.seconds:.2f} | {len(c.text)} | {split} |")
    return "\n".join(lines) + "\n"
