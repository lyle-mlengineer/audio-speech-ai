"""Stage 2: parquet -> 24 kHz WAVs + an F5-TTS dataset + a reference clip.

The dataset is small (42 rows, 21 min, one speaker) and awkward in two ways:

* every clip is exactly 30.000 s at 22.05 kHz, cut on a grid, so some cuts land
  mid-word. We keep the transcript verbatim anyway - rewriting it is what
  misaligns text and audio.
* there are no word timestamps, so the ~9 s reference clip for voice cloning
  has to be cut at a silence and its transcript estimated from character share.
  The estimate is written to ``ref_meta.json`` and played in the notebook so it
  can be checked by ear.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import (
    DATASET_ID,
    DATASET_PARQUET,
    MODEL_REPO_ID,
    SAMPLE_RATE,
    SOURCE_SAMPLE_RATE,
    VOCAB_FILE,
    Paths,
)
from .resample import resample, to_mono
from .text import (
    charset_report,
    choose_ref_prefix,
    clean_text,
    estimate_fraction,
    f5_text_identity,
    read_vocab,
)

#: F5's CustomDataset accepts 0.3 <= duration <= 30. Keep a hair under 30 so a
#: float rounding error cannot make the whole sample disappear.
MAX_TRAIN_SECONDS = 29.98


# --------------------------------------------------------------------- download
def download_parquet(paths: Paths, force: bool = False) -> Path:
    if paths.raw_parquet.is_file() and not force:
        return paths.raw_parquet
    from huggingface_hub import hf_hub_download

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=DATASET_PARQUET,
        local_dir=str(paths.raw_parquet.parent),
    )
    local = Path(local)
    if local.resolve() != paths.raw_parquet.resolve():
        paths.raw_parquet.write_bytes(local.read_bytes())
    return paths.raw_parquet


def ensure_vocab(paths: Paths, force: bool = False) -> Path:
    """Download Sauti's vocab.txt once, and copy it into the F5 dataset dir.

    This is *not* optional. F5's own prepare script copies the Emilia vocab; if
    that were used with a Sauti checkpoint, every token index would be wrong.
    The checkpoint's text embedding and this file must have the same length.
    """
    if force or not paths.sauti_vocab.is_file():
        from huggingface_hub import hf_hub_download

        paths.pretrained_dir.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=VOCAB_FILE,
                local_dir=str(paths.pretrained_dir),
            )
        )
        if downloaded.resolve() != paths.sauti_vocab.resolve():
            paths.sauti_vocab.write_bytes(downloaded.read_bytes())

    paths.f5_dataset_dir.mkdir(parents=True, exist_ok=True)
    if force or not paths.f5_vocab.is_file():
        paths.f5_vocab.write_bytes(paths.sauti_vocab.read_bytes())
    return paths.sauti_vocab


# ------------------------------------------------------------------- audio I/O
def read_rows(parquet: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(parquet).to_pylist()


def decode_clip(audio_field) -> tuple[np.ndarray, int]:
    """Decode the HF ``audio`` struct (bytes or path) to mono float32."""
    import io

    import soundfile as sf

    if isinstance(audio_field, dict) and audio_field.get("bytes") is not None:
        x, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32", always_2d=False)
    elif isinstance(audio_field, dict) and audio_field.get("path"):
        x, sr = sf.read(audio_field["path"], dtype="float32", always_2d=False)
    else:
        raise ValueError("unsupported audio field; expected {'bytes': ...} or {'path': ...}")
    return to_mono(x), int(sr)


def peak_normalise(x: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > ceiling:
        return (x * (ceiling / peak)).astype(np.float32)
    return x.astype(np.float32)


# --------------------------------------------------------------------- silence
def frame_rms(x: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> tuple[np.ndarray, float]:
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if x.size < frame:
        return np.zeros(1, dtype=np.float32), hop_ms / 1000.0
    energy = np.convolve(x.astype(np.float64) ** 2, np.ones(frame) / frame, mode="valid")
    return np.sqrt(energy[::hop]).astype(np.float32), hop / sr


def find_silence_cut(
    x: np.ndarray,
    sr: int,
    target_seconds: float,
    search_seconds: float = 2.0,
    min_silence_seconds: float = 0.08,
) -> tuple[float, float] | None:
    """Find the quietest pause near ``target_seconds``.

    Returns ``(cut_seconds, silence_seconds)`` or None. Cutting mid-word is the
    main way to poison a reference clip, so a real pause is preferred; if there
    is none the caller falls back to the target and marks confidence low.
    """
    rms, hop_s = frame_rms(x, sr)
    if rms.size < 3:
        return None
    # ``percentile(rms, 5)`` is the noise floor, not ``percentile(..., 20)``:
    # conversational clips can have less than 20% silence, and then the 20th
    # percentile is *speech*, which would mark every frame as quiet.
    noise = float(np.percentile(rms, 5))
    loud = float(np.percentile(rms, 95))
    threshold = max(1e-5, noise * 1.5, loud * 0.05)
    threshold = min(threshold, loud * 0.3)  # never call normal speech "silence"
    quiet = rms < threshold

    # Runs of quiet frames.
    edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet.view(np.int8), [0]))))
    runs = []
    for start, end in zip(edges[0::2], edges[1::2]):
        if end - start < 2:
            continue
        start_s = start * hop_s
        end_s = end * hop_s
        runs.append((start_s, end_s, end_s - start_s))

    candidates = []
    for start_s, end_s, length in runs:
        if length < min_silence_seconds:
            continue
        centre = 0.5 * (start_s + end_s)
        if abs(centre - target_seconds) > search_seconds:
            continue
        score = min(length, 0.5) - 0.15 * abs(centre - target_seconds)
        candidates.append((score, centre, length))
    if not candidates:
        return None
    _, centre, length = max(candidates)
    return float(centre), float(length)


def apply_fades(x: np.ndarray, sr: int, fade_ms: float = 10.0) -> np.ndarray:
    n = max(1, int(sr * fade_ms / 1000.0))
    n = min(n, x.size // 2)
    if n <= 0:
        return x
    y = x.copy()
    y[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return y


# ------------------------------------------------------------------ splitting
def split_indices(n: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    if n <= 1:
        return list(range(n)), []
    val_size = min(max(0, val_size), n - 1)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return sorted(order[val_size:]), sorted(order[:val_size])


# ------------------------------------------------------------------ reference
@dataclass(frozen=True)
class ReferenceChoice:
    source_index: int
    utt_id: str
    cut_seconds: float
    silence_seconds: float | None
    text: str
    text_fraction: float
    confidence: str

    def as_dict(self) -> dict:
        return {
            "source_index": self.source_index,
            "utt_id": self.utt_id,
            "cut_seconds": round(self.cut_seconds, 3),
            "silence_seconds": None if self.silence_seconds is None else round(self.silence_seconds, 3),
            "text_fraction": round(self.text_fraction, 4),
            "confidence": self.confidence,
            "ref_text": self.text,
        }


def choose_reference(
    candidates: list[dict],
    target_seconds: float = 9.0,
) -> tuple[ReferenceChoice, np.ndarray]:
    """Pick the val clip whose pause nearest ``target_seconds`` is cleanest.

    ``candidates`` items: ``{index, utt_id, audio (np), sr, text, duration}``.
    """
    best: tuple[float, ReferenceChoice, np.ndarray] | None = None
    for item in candidates:
        x = item["audio"]
        sr = item["sr"]
        total = len(x) / sr
        target = min(target_seconds, total * 0.6)
        found = find_silence_cut(x, sr, target)
        if found is not None:
            cut, silence = found
            confidence = "silence" if silence >= 0.12 else "short-silence"
        else:
            cut, silence = target, None
            confidence = "no-silence"

        prefix = choose_ref_prefix(item["text"], cut / total)
        mismatch = abs(prefix.fraction - cut / total)
        # A long pause is good, a text/audio mismatch is bad, being near the
        # target duration is mildly good.
        score = 2.0 * (silence or 0.0) - 3.0 * mismatch - 0.05 * abs(cut - target_seconds)
        if best is None or score > best[0]:
            choice = ReferenceChoice(
                source_index=int(item["index"]),
                utt_id=str(item["utt_id"]),
                cut_seconds=cut,
                silence_seconds=silence,
                text=prefix.text,
                text_fraction=prefix.fraction,
                confidence=confidence,
            )
            best = (score, choice, x[: int(cut * sr)])

    if best is None:
        raise RuntimeError("no validation clips available to build a reference")
    _, choice, audio = best
    return choice, audio


# --------------------------------------------------------------------- outputs
def write_metadata_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for row in rows:
            writer.writerow([row["audio_path"], row["text"]])


def write_f5_dataset(records: list[dict], durations: list[float], paths: Paths) -> None:
    from datasets.arrow_writer import ArrowWriter

    paths.f5_dataset_dir.mkdir(parents=True, exist_ok=True)
    with ArrowWriter(path=str(paths.f5_raw_arrow)) as writer:
        for record in records:
            writer.write(record)
        writer.finalize()
    paths.f5_duration_json.write_text(
        json.dumps({"duration": durations}, indent=2) + "\n", encoding="utf-8"
    )


def _write_report(
    paths: Paths,
    stats: dict,
    rows: list[dict],
    charset,
    ref: ReferenceChoice,
    warnings: list[str],
) -> None:
    lines = [
        "# Data report",
        "",
        f"- dataset: `{DATASET_ID}` (speaker `{stats['speaker']}`)",
        f"- clips: **{stats['n_clips']}**, total **{stats['total_seconds'] / 60:.2f} min**",
        f"- source: {stats['source_sample_rate']} Hz -> {stats['sample_rate']} Hz, mono",
        f"- train / val: **{stats['n_train']} / {stats['n_val']}**",
        f"- characters: {stats['chars_per_second']:.2f} chars/s",
        "",
        "## Transcript characters vs Sauti vocab",
        "",
        f"- unique characters: {charset.unique_chars}",
        f"- missing from vocab: **{len(charset.missing)}**"
        + (f" -> `{charset.missing}`" if charset.missing else ""),
        f"- non-ASCII: {len(charset.non_ascii)}",
        f"- digits: {charset.digits} (left as-is; F5 pronounces digits poorly, but"
        " rewriting them would break the transcript/audio pair)",
        f"- apostrophes: {charset.apostrophes} (kept: `ng'ang'ana` is not `ngangana`)",
        f"- F5 `convert_char_to_pinyin`: {stats.get('pinyin_identity', 'unknown')}",
        "",
        "## Reference clip",
        "",
        f"- source: `{ref.utt_id}` (index {ref.source_index})",
        f"- cut: **{ref.cut_seconds:.2f} s**, pause: {ref.silence_seconds}, confidence: **{ref.confidence}**",
        f"- transcript estimated from {ref.text_fraction * 100:.1f}% of the clip's characters",
        f"- ref text: `{ref.text}`",
        "",
        "> The reference transcript is an estimate, not a forced alignment. Play"
        " `ref_audio.wav` and read `ref_text.txt` before trusting the eval: a"
        " wrong ref text mainly costs timbre fidelity, not intelligibility.",
        "",
    ]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    paths.data_report.write_text("\n".join(lines), encoding="utf-8")


def prepare(
    paths: Paths,
    val_size: int = 4,
    seed: int = 42,
    ref_index: int | None = None,
    ref_seconds: float = 9.0,
    force: bool = False,
) -> dict:
    import soundfile as sf

    paths.ensure()
    if paths.dataset_info.is_file() and not force:
        try:
            return json.loads(paths.dataset_info.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    ensure_vocab(paths)
    parquet = download_parquet(paths, force=force)
    rows = read_rows(parquet)
    if not rows:
        raise RuntimeError(f"no rows in {parquet}")

    vocab = read_vocab(paths.sauti_vocab)
    texts: list[str] = []
    durations: list[float] = []
    records: list[dict] = []
    warnings: list[str] = []
    source_rates: set[int] = set()
    clipped = 0

    for i, row in enumerate(rows):
        utt_id = f"utt_{i + 1:04d}"
        audio, sr = decode_clip(row.get("audio"))
        source_rates.add(sr)
        if sr != SAMPLE_RATE:
            audio = resample(audio, sr, SAMPLE_RATE)
        if audio.size and float(np.max(np.abs(audio))) > 0.99:
            clipped += 1
        audio = peak_normalise(audio)

        duration = audio.size / SAMPLE_RATE
        if duration > MAX_TRAIN_SECONDS:
            audio = audio[: int(MAX_TRAIN_SECONDS * SAMPLE_RATE)]
            duration = audio.size / SAMPLE_RATE

        text = clean_text(row.get("text", ""))
        if not text:
            warnings.append(f"{utt_id}: empty transcript, skipped")
            continue

        wav_path = paths.wav_dir / f"{utt_id}.wav"
        sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")

        texts.append(text)
        durations.append(duration)
        records.append(
            {
                "audio_path": str(wav_path.resolve()),
                "text": text,
                "duration": float(duration),
            }
        )

    if not records:
        raise RuntimeError("no usable rows after filtering")

    train_idx, val_idx = split_indices(len(records), val_size, seed)
    train_rows = [records[i] for i in train_idx]
    val_rows = [records[i] for i in val_idx]
    n_val_total = len(val_rows)
    write_metadata_csv(train_rows, paths.train_csv)
    write_metadata_csv(val_rows, paths.val_csv)
    write_f5_dataset(records, durations, paths)

    # Reference: only from validation so the fine-tune never hears it.
    if ref_index is not None:
        if not (0 <= ref_index < len(val_rows)):
            raise ValueError(f"--ref-index {ref_index} out of range for {len(val_rows)} val clips")
        val_rows = [val_rows[ref_index]]
    candidates = []
    for row in val_rows:
        audio, sr = sf.read(row["audio_path"], dtype="float32", always_2d=False)
        candidates.append(
            {
                "index": int(Path(row["audio_path"]).stem.split("_")[-1]) - 1,
                "utt_id": Path(row["audio_path"]).stem,
                "audio": to_mono(audio),
                "sr": sr,
                "text": row["text"],
                "duration": row["duration"],
            }
        )
    ref, ref_audio = choose_reference(candidates, target_seconds=ref_seconds)
    sf.write(str(paths.ref_audio), apply_fades(ref_audio, SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_16")
    paths.ref_text_file.write_text(ref.text + "\n", encoding="utf-8")
    paths.ref_meta.write_text(json.dumps(ref.as_dict(), indent=2) + "\n", encoding="utf-8")

    charset = charset_report(texts, vocab)
    identity = [f5_text_identity(t) for t in texts]
    changed = sum(1 for value in identity if value is False)
    if changed:
        warnings.append(
            f"{changed} transcript(s) are changed by F5's convert_char_to_pinyin; "
            "training stores the verbatim text, so inference would tokenise them differently"
        )
    if charset.missing:
        warnings.append(
            "transcript characters missing from the Sauti vocab (they become index 0 = space): "
            + str(charset.missing)
        )
    if source_rates != {SOURCE_SAMPLE_RATE}:
        warnings.append(f"unexpected source sample rates: {sorted(source_rates)}")
    if clipped:
        warnings.append(f"{clipped} clips peaked above 0.99 and were scaled down")
    if any(abs(d - 30.0) < 1e-6 for d in durations):
        warnings.append(
            "clips are exactly 30.0 s (cut on a grid); some may start/end mid-word. "
            "Transcripts were left verbatim."
        )

    stats = {
        "dataset": DATASET_ID,
        "speaker": rows[0].get("speaker", "unknown") if rows else "unknown",
        "n_clips": len(records),
        "n_train": len(train_rows),
        "n_val": n_val_total,
        "total_seconds": float(sum(durations)),
        "source_sample_rate": sorted(source_rates)[0] if len(source_rates) == 1 else sorted(source_rates),
        "sample_rate": SAMPLE_RATE,
        "chars_per_second": float(sum(len(t) for t in texts) / max(sum(durations), 1e-9)),
        "pinyin_identity": (
            "not checked (f5-tts not installed)"
            if not any(value is not None for value in identity)
            else "identity on every transcript"
            if changed == 0
            else f"{changed} transcript(s) changed"
        ),
        "train_csv": str(paths.train_csv),
        "val_csv": str(paths.val_csv),
        "f5_dataset_dir": str(paths.f5_dataset_dir),
        "vocab": str(paths.sauti_vocab),
        "vocab_size": len(vocab),
        "reference": ref.as_dict(),
        "warnings": warnings,
    }
    paths.dataset_info.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    _write_report(paths, stats, records, charset, ref, warnings)
    return stats
