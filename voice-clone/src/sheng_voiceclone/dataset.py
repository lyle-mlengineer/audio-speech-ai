"""Stage 2: the dataset -> 16 kHz WAVs, splits, manifests and one reference clip.

The Hugging Face dataset ``lyle-mlengineer/kenyan-celebs`` is a single speaker
(``abel``) talking Sheng: 42 clips of 30.0 s, 22.05 kHz 16-bit mono, with
transcript. Three facts drive this stage:

* every clone model here consumes **16 kHz mono**, so audio is converted once;
* the **reference clip must not be in the training split**, otherwise the tuned
  model can memorise it and the speaker-similarity score becomes meaningless;
* transcripts are used verbatim. Every clip is exactly 30 s because it was cut on
  a grid, so some cuts land mid-word - rewriting the text is what misaligns
  text and audio. Leave them alone.

No ``datasets`` dependency: pyarrow reads the parquet, soundfile decodes it.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import manifest as mf
from .audio import resample, to_mono, write_wav
from .paths import DATASET_ID, DATASET_PARQUET, SAMPLE_RATE, SPEAKER, Paths

MIN_SECONDS = 1.0


@dataclass
class Clip:
    index: int
    text: str
    speaker: str
    seconds: float
    wav: Path


# ------------------------------------------------------------------- download
def download_parquet(paths: Paths, *, force: bool = False) -> Path:
    """Fetch the single 52 MB parquet into the artifacts root."""
    dest = paths.raw_dir / Path(DATASET_PARQUET).name
    if dest.is_file() and not force and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        got = hf_hub_download(
            repo_id=DATASET_ID,
            filename=DATASET_PARQUET,
            repo_type="dataset",
            local_dir=str(paths.raw_dir),
        )
        src = Path(got)
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
    except Exception:  # no huggingface_hub / no cache: plain HTTPS
        from urllib.request import urlopen

        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/{DATASET_PARQUET}"
        with urlopen(url) as r, dest.open("wb") as f:
            f.write(r.read())  # type: ignore[arg-type]
    if not dest.is_file() or dest.stat().st_size < 1_000_000:
        raise RuntimeError(f"download failed or truncated: {dest}")
    return dest


# ---------------------------------------------------------------------- decode
def read_clips(parquet: Path, wav_dir: Path, *, force: bool = False) -> list[Clip]:
    """Decode every row, resample to 16 kHz, write one PCM16 WAV per row."""
    import pyarrow.parquet as pq
    import soundfile as sf

    rows = pq.read_table(parquet).to_pylist()
    wav_dir.mkdir(parents=True, exist_ok=True)

    clips: list[Clip] = []
    for i, row in enumerate(rows, start=1):
        wav_path = wav_dir / f"utt_{i:04d}.wav"
        text = str(row["text"]).strip()
        speaker = str(row.get("speaker") or SPEAKER)
        if wav_path.is_file() and not force:
            secs = float(sf.info(str(wav_path)).duration)
        else:
            x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
            x = to_mono(x)
            if int(sr) != SAMPLE_RATE:
                x = resample(x, int(sr), SAMPLE_RATE)
            write_wav(wav_path, np.clip(x, -1.0, 1.0), SAMPLE_RATE)
            secs = float(sf.info(str(wav_path)).duration)
        clips.append(Clip(index=i, text=text, speaker=speaker, seconds=float(secs), wav=wav_path))
    return [c for c in clips if c.seconds >= MIN_SECONDS]


def read_dir(src: Path, wav_dir: Path, *, force: bool = False) -> list[Clip]:
    """Use your own recordings: ``src/*.wav`` with a ``.txt`` sidecar or ``src.csv``."""
    src = Path(src)
    texts: dict[str, str] = {}
    csv = src / "transcripts.csv"
    if csv.is_file():
        import csv as _csv

        with csv.open("r", encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                texts[str(row["file"]).strip()] = str(row["text"]).strip()
    clips: list[Clip] = []
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a"})
    for i, p in enumerate(files, start=1):
        text = texts.get(p.name, "")
        if not text and p.with_suffix(".txt").is_file():
            text = p.with_suffix(".txt").read_text(encoding="utf-8").strip()
        if not text:
            raise RuntimeError(f"{p.name}: no transcript (add {p.stem}.txt or transcripts.csv)")
        out = wav_dir / f"utt_{i:04d}.wav"
        import soundfile as sf

        if out.is_file() and not force:
            secs = float(sf.info(str(out)).duration)
        else:
            x, sr = sf.read(str(p), dtype="float32", always_2d=False)
            x = to_mono(x)
            if int(sr) != SAMPLE_RATE:
                x = resample(x, int(sr), SAMPLE_RATE)
            write_wav(out, np.clip(x, -1.0, 1.0), SAMPLE_RATE)
            secs = float(sf.info(str(out)).duration)
        clips.append(Clip(index=i, text=text, speaker=SPEAKER, seconds=float(secs), wav=out))
    return [c for c in clips if c.seconds >= MIN_SECONDS]


# ----------------------------------------------------------------------- build
def split(
    clips: list[Clip], *, val_clips: int = 4, test_clips: int = 4, seed: int = 1337
) -> tuple[list[Clip], list[Clip], list[Clip]]:
    """Shuffle then cut val/test off the end; the rest is train."""
    if len(clips) < val_clips + test_clips + 4:
        raise RuntimeError(f"need >= {val_clips + test_clips + 4} clips, have {len(clips)}")
    order = list(clips)
    random.Random(seed).shuffle(order)
    val = order[:val_clips]
    test = order[val_clips : val_clips + test_clips]
    train = order[val_clips + test_clips :]
    return train, val, test


def pick_reference(val: list[Clip]) -> Clip:
    """Longest validation clip: most of the speaker in one file, none of it in train."""
    return max(val, key=lambda c: c.seconds)


def _row(c: Clip, paths: Paths) -> dict[str, object]:
    return {
        "audio": mf.rel_audio(c.wav, paths.root),
        "text": c.text,
        "speaker": c.speaker,
        "language": "sw",
        "seconds": round(c.seconds, 3),
    }


def build(
    paths: Paths,
    *,
    source: str | None = None,
    parquet: str | Path | None = None,
    val_clips: int = 4,
    test_clips: int = 4,
    seed: int = 1337,
    force: bool = False,
) -> dict[str, object]:
    """Run the whole stage.

    ``source`` = a local directory instead of the dataset; ``parquet`` = an
    already-downloaded parquet (used by tests and by a warmed Drive cache).
    """
    paths.ensure_dirs()
    if source:
        clips = read_dir(Path(source).expanduser(), paths.wav_dir, force=force)
        origin = f"local dir {source}"
    else:
        pq_path = Path(parquet) if parquet else download_parquet(paths, force=force)
        clips = read_clips(pq_path, paths.wav_dir, force=force)
        origin = str(pq_path)
    if not clips:
        raise RuntimeError("no usable clips after decoding")

    train, val, test = split(clips, val_clips=val_clips, test_clips=test_clips, seed=seed)
    ref = pick_reference(val)

    # Copy the reference to one canonical path: every consumer points at it.
    import shutil

    shutil.copyfile(ref.wav, paths.ref_audio)
    paths.ref_text.write_text(ref.text + "\n", encoding="utf-8")

    mf.write_jsonl(paths.train_manifest, [_row(c, paths) for c in train])
    mf.write_jsonl(paths.val_manifest, [_row(c, paths) for c in val])
    mf.write_jsonl(
        paths.test_manifest,
        [{**_row(c, paths), "is_reference": c.wav == ref.wav} for c in test],
    )

    info: dict[str, object] = {
        "source": origin,
        "clips": len(clips),
        "speaker": clips[0].speaker,
        "sample_rate": SAMPLE_RATE,
        "minutes": round(sum(c.seconds for c in clips) / 60.0, 1),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": seed,
        "ref_audio": paths.rel(paths.ref_audio),
        "ref_seconds": round(ref.seconds, 1),
        "ref_text": ref.text,
    }
    (paths.data_dir / "data_report.md").write_text(render_report(info, paths), encoding="utf-8")
    (paths.data_dir / "data_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return info


def render_report(info: dict[str, object], paths: Paths) -> str:
    lines = [
        "# Sheng voice-clone data",
        "",
        f"* source: `{info['source']}`",
        f"* speaker: **{info['speaker']}**, {info['clips']} clips, {info['minutes']} min @ {info['sample_rate']} Hz",
        f"* split: {info['train']} train / {info['val']} val / {info['test']} test (seed {info['seed']})",
        f"* reference clip: `{info['ref_audio']}` ({info['ref_seconds']} s, from **val** so it is never trained on)",
        "",
        "## Reference transcript",
        "",
        f"> {info['ref_text']}",
        "",
        "## Files",
        "",
        f"* `{paths.rel(paths.train_manifest)}` - what the trainer reads",
        f"* `{paths.rel(paths.val_manifest)}` - held-out, for listening during training",
        f"* `{paths.rel(paths.test_manifest)}` - never trained on; `compare` scores against it",
        "",
    ]
    if str(info["source"]).startswith(DATASET_ID):
        lines += [
            "Transcripts are verbatim. Clips were cut on a 30 s grid, so a few start or",
            "end mid-word; that is a property of the source, not a bug to fix here.",
            "",
        ]
    return "\n".join(lines)
