"""Build caption-conditioned training data for Kenyan voice design.

Voice design in VoxCPM2 is prompt-conditioned generation: the description sits in
parentheses in front of the text (``"(A warm Kenyan male voice)Habari yako?"``).
Fine-tuning therefore needs rows whose ``text`` is *caption + transcript*, paired
with audio from the speaker that caption describes. That is the whole trick: the
LoRA learns to bind "Kenyan, female, thirties, mid-pitched" to real Kenyan
speech, and at inference the same kind of sentence designs a new voice.

Pipeline (two passes over the HF dataset, both streamed - the source is ~44 GB):

1. **Metadata pass** (parquet columns only, no audio decoded): per speaker
   counts, durations, gender / age / location votes. Speakers are then selected
   deterministically (hash order) and split **by speaker**, so validation voices
   are never seen during training.
2. **Audio pass** (filtered to the selected clip ids): resample to 16 kHz mono
   PCM16, measure median f0 per clip, write WAVs.

Captions are then built from aggregated *speaker-level* attributes (see
``captions.build_profile``): a voice-design attribute belongs to a speaker, so
all of one speaker's rows carry the same description and the model never has to
reconcile "male" with a female-sounding clip.

Everything lands under ``paths.artifacts`` and is idempotent; a rerun with the
same config reuses the WAVs instead of streaming 20 GB again.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import captions as C
from . import manifest as M
from .audio_features import median_f0
from .paths import DATASET_PRESETS, DEFAULT_DATASETS, TARGET_SAMPLE_RATE, Paths

_WS = re.compile(r"\s+")

# Metadata columns we know how to use, in the order they are requested.
META_KEYS = ("id", "speaker", "duration", "gender", "age", "location", "text")


@dataclass
class PrepareConfig:
    preset: str = DEFAULT_DATASETS[0]
    dataset_id: str = ""  # overrides the preset repo
    split: str = ""  # overrides the preset split
    language: str = ""  # overrides the preset language
    target_sr: int = TARGET_SAMPLE_RATE
    max_speakers: int = 120
    max_clips_per_speaker: int = 30
    min_clips_per_speaker: int = 6
    max_audio_hours: float = 10.0
    max_scan_rows: int = 60_000
    min_dur: float = 3.0
    max_dur: float = 20.0
    val_speaker_fraction: float = 0.1
    val_clips_per_speaker: int = 3
    caption_variants: int = 1
    caption_dropout: float = 0.1
    seed: int = 1337
    force: bool = False

    def preset_values(self) -> dict[str, str]:
        if self.preset not in DATASET_PRESETS:
            raise ValueError(f"unknown dataset preset {self.preset!r}; have {sorted(DATASET_PRESETS)}")
        return dict(DATASET_PRESETS[self.preset])

    @property
    def resolved_dataset(self) -> str:
        return self.dataset_id or self.preset_values()["repo"]

    @property
    def resolved_split(self) -> str:
        return self.split or self.preset_values().get("split") or "train"

    @property
    def resolved_language(self) -> str:
        return self.language or self.preset_values().get("language") or "Swahili"

    def columns(self) -> dict[str, str]:
        """Column names for this preset (``""`` = the dataset does not have it)."""
        preset = self.preset_values()
        return {
            "audio": preset.get("audio") or "audio",
            "text": preset.get("text") or "text",
            "speaker": preset.get("speaker") or "",
            "id": preset.get("id") or "",
            "duration": preset.get("duration") or "",
            "gender": preset.get("gender") or "",
            "age": preset.get("age") or "",
            "location": preset.get("location") or "",
        }

    def meta_columns(self) -> tuple[str, ...]:
        cols = self.columns()
        names = [cols[key] for key in META_KEYS if cols[key]]
        if not cols["duration"]:
            names.append(cols["audio"])  # duration has to be measured from the audio
        return tuple(dict.fromkeys(names))

    def fingerprint(self) -> dict[str, Any]:
        """Everything that changes the produced WAVs/manifests."""
        return {
            "dataset": self.resolved_dataset,
            "split": self.resolved_split,
            "language": self.resolved_language,
            "columns": self.columns(),
            "target_sr": self.target_sr,
            "max_speakers": self.max_speakers,
            "max_clips_per_speaker": self.max_clips_per_speaker,
            "min_clips_per_speaker": self.min_clips_per_speaker,
            "max_audio_hours": round(self.max_audio_hours, 3),
            "min_dur": self.min_dur,
            "max_dur": self.max_dur,
            "val_speaker_fraction": self.val_speaker_fraction,
            "val_clips_per_speaker": self.val_clips_per_speaker,
            "caption_variants": self.caption_variants,
            "caption_dropout": self.caption_dropout,
            "seed": self.seed,
        }


# ------------------------------------------------------------------ helpers
def clean_text(text: str) -> str:
    """Conservative normalisation: transcripts must stay word-for-word faithful."""
    text = (text or "").replace("\u00a0", " ").replace("\r", " ").replace("\n", " ")
    return _WS.sub(" ", text).strip()


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", (name or "speaker")).strip("-").lower() or "speaker"


def _hash_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _stream(dataset_id: str, split: str, columns: Iterable[str] | None = None):
    """Streamed HF dataset; ``columns`` keeps the metadata pass from decoding audio."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if columns:
        kwargs["columns"] = list(columns)
    try:
        return load_dataset(dataset_id, **kwargs)
    except TypeError:  # older datasets: no `columns` for streaming
        kwargs.pop("columns", None)
        return load_dataset(dataset_id, **kwargs)


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
    except Exception:  # noqa: BLE001 - torchaudio kernel may be missing
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


def wav_path_for(paths: Paths, speaker: str, audio_id: str) -> Path:
    """Stable WAV name: derived from the clip id, so rebuilds never mix clips up."""
    return paths.wav_dir / f"{_slug(speaker)}_{_slug(audio_id)[:24]}.wav"


def _mode(values: list[str]) -> str:
    vals = [v for v in values if v]
    return Counter(vals).most_common(1)[0][0] if vals else ""


# ----------------------------------------------------------- metadata pass
@dataclass
class SpeakerScan:
    speaker: str
    clips: list[tuple[str, float]] = field(default_factory=list)  # (audio_id, duration)
    genders: list[str] = field(default_factory=list)
    ages: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(d for _, d in self.clips)


def scan_metadata(cfg: PrepareConfig, *, log=print) -> tuple[dict[str, SpeakerScan], dict[str, str], int]:
    """One streaming pass over metadata columns (audio only if duration is absent)."""
    dataset_id, split = cfg.resolved_dataset, cfg.resolved_split
    cols = cfg.columns()
    measure_duration = not cols["duration"]
    log(
        f"[data] scanning metadata of {dataset_id} [{split}]"
        + (" (decoding audio: no duration column)" if measure_duration else " (no audio)")
    )
    ds = _stream(dataset_id, split, columns=cfg.meta_columns())

    scans: dict[str, SpeakerScan] = {}
    texts: dict[str, str] = {}
    rows = skipped = 0
    default_speaker = dataset_id.split("/")[-1] or "speaker"
    for row in ds:
        rows += 1
        text = clean_text(row.get(cols["text"]) or "")
        speaker = str(row.get(cols["speaker"]) or default_speaker) if cols["speaker"] else default_speaker
        audio_id = str(row.get(cols["id"]) or f"row{rows}") if cols["id"] else f"row{rows}"
        if measure_duration:
            audio = row[cols["audio"]]
            dur = float(len(audio["array"])) / float(audio["sampling_rate"])
        else:
            dur = float(row.get(cols["duration"]) or 0.0)
        if not text or not speaker or not (cfg.min_dur <= dur <= cfg.max_dur):
            skipped += 1
            continue
        scan = scans.setdefault(speaker, SpeakerScan(speaker=speaker))
        scan.clips.append((audio_id, dur))
        scan.genders.append(str(row.get(cols["gender"]) or "") if cols["gender"] else "")
        scan.ages.append(str(row.get(cols["age"]) or "") if cols["age"] else "")
        scan.locations.append(str(row.get(cols["location"]) or "") if cols["location"] else "")
        texts[audio_id] = text

    log(f"[data] metadata: {rows} rows, {len(scans)} usable speakers, {skipped} skipped")
    if len(scans) == 1:
        log("[data] WARNING: one speaker only - this can shift the accent but cannot teach voice design")
    return scans, texts, rows


def select_speakers(
    scans: dict[str, SpeakerScan], cfg: PrepareConfig, *, log=print
) -> tuple[list[str], list[str]]:
    """Deterministic speaker selection + speaker-disjoint train/val split."""
    eligible = [s for s in scans.values() if len(s.clips) >= cfg.min_clips_per_speaker]
    if not eligible:
        raise RuntimeError(
            f"no speaker has >= {cfg.min_clips_per_speaker} clips in the {cfg.min_dur}-{cfg.max_dur}s "
            "window; lower --min-clips-per-speaker"
        )
    # Hash order: stable across machines and independent of dataset ordering.
    eligible.sort(key=lambda s: _hash_key(f"{cfg.seed}:{s.speaker}"))

    chosen: list[str] = []
    budget = cfg.max_audio_hours * 3600
    seconds = 0.0
    for scan in eligible:
        if len(chosen) >= cfg.max_speakers or seconds >= budget:
            break
        chosen.append(scan.speaker)
        seconds += scan.seconds

    n_val = 0
    if len(chosen) >= 4 and cfg.val_speaker_fraction > 0:
        n_val = max(2, round(cfg.val_speaker_fraction * len(chosen)))
        n_val = min(n_val, len(chosen) - 2)
    val = sorted(chosen[:n_val], key=lambda s: _hash_key(f"val:{s}"))
    train = [s for s in chosen if s not in set(val)]
    log(
        f"[data] selected {len(chosen)} speakers ({len(train)} train / {len(val)} val), "
        f"{sum(scans[s].seconds for s in chosen) / 3600:.1f} h available"
    )
    return train, val


def pick_clips(scans: dict[str, SpeakerScan], speakers: list[str], cfg: PrepareConfig, cap: int, salt: str) -> dict[str, list[str]]:
    """Deterministic per-speaker clip ids (hash-shuffled, capped)."""
    picked: dict[str, list[str]] = {}
    for speaker in speakers:
        ids = [audio_id for audio_id, _ in scans[speaker].clips]
        ids.sort(key=lambda i: _hash_key(f"{cfg.seed}:{salt}:{i}"))
        picked[speaker] = ids[:cap]
    return picked


# -------------------------------------------------------------- audio pass
def fetch_audio(
    paths: Paths,
    scans: dict[str, SpeakerScan],
    texts: dict[str, str],
    wanted: dict[str, list[str]],
    cfg: PrepareConfig,
    *,
    log=print,
) -> dict[str, dict[str, Any]]:
    """Second streaming pass, filtered to the selected clip ids.

    The filter uses ``input_columns=["audio_id"]`` so VoxCPM-sized audio is only
    fetched for clips we actually keep. WAV names are stable, so a rerun skips
    files that already exist on Drive.
    """
    wanted_ids = {audio_id for ids in wanted.values() for audio_id in ids}
    owner = {audio_id: speaker for speaker, ids in wanted.items() for audio_id in ids}
    done: dict[str, dict[str, Any]] = {}

    # Reuse WAVs already on Drive instead of streaming them again.
    for audio_id in list(wanted_ids):
        speaker = owner[audio_id]
        wav = wav_path_for(paths, speaker, audio_id)
        if wav.exists():
            done[audio_id] = {
                "audio": paths.rel_to_manifest(wav),
                "duration": M.audio_duration(wav),
                "speaker": speaker,
                "f0_hz": None,
                "text": texts.get(audio_id, ""),
            }
    if len(done) == len(wanted_ids) and wanted_ids:
        log(f"[data] all {len(done)} wavs already present, skipping the audio pass")
        return done

    dataset_id, split = cfg.resolved_dataset, cfg.resolved_split
    log(f"[data] streaming audio for {len(wanted_ids) - len(done)} clips from {dataset_id} [{split}]")
    ds = _stream(dataset_id, split)

    id_column = cfg.columns()["id"] or "audio_id"

    def keep(example: dict[str, Any]) -> bool:  # noqa: ANN001 - datasets callback
        return str(example.get(id_column, "")) in wanted_ids

    try:
        ds = ds.filter(keep, input_columns=[id_column])
    except TypeError:  # older datasets: filter() without input_columns
        ds = ds.filter(keep)

    cols = cfg.columns()
    scanned = written = failed = 0
    for row in ds:
        scanned += 1
        audio_id = str(row.get(cols["id"]) or "") if cols["id"] else ""
        if not audio_id:
            audio_id = f"row{scanned}"
        if audio_id in done:
            continue
        try:
            audio = row[cols["audio"]]
            array = audio["array"]
            src_sr = int(audio["sampling_rate"])
            arr16 = _resample(array, src_sr, cfg.target_sr)
            speaker = owner[audio_id]
            wav = wav_path_for(paths, speaker, audio_id)
            dur = _write_wav(wav, arr16, cfg.target_sr)
            f0 = median_f0(arr16, cfg.target_sr)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the build
            failed += 1
            log(f"[data] skip {audio_id}: {type(exc).__name__}: {exc}")
            continue
        if dur < cfg.min_dur:
            wav.unlink(missing_ok=True)
            continue
        done[audio_id] = {
            "audio": paths.rel_to_manifest(wav),
            "duration": round(dur, 3),
            "speaker": speaker,
            "f0_hz": f0,
            "text": texts.get(audio_id, ""),
        }
        written += 1
        if written % 100 == 0:
            log(f"[data] {written} wavs written ({scanned} rows visited)")
        if len(done) >= len(wanted_ids):
            break
        if scanned >= cfg.max_scan_rows:
            log(f"[data] scan limit {cfg.max_scan_rows} rows reached - continuing with what we have")
            break

    log(f"[data] audio pass: {written} new wavs, {failed} failed, {scanned} rows visited")
    return done


# ------------------------------------------------------------------- build
def build_profiles(
    scans: dict[str, SpeakerScan],
    clips: dict[str, dict[str, Any]],
    cfg: PrepareConfig,
) -> dict[str, C.SpeakerProfile]:
    """One captionable profile per speaker, from metadata + measured f0/pace."""
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for info in clips.values():
        by_speaker.setdefault(info["speaker"], []).append(info)

    profiles: dict[str, C.SpeakerProfile] = {}
    for speaker, items in by_speaker.items():
        scan = scans[speaker]
        f0_values = [i["f0_hz"] for i in items if i.get("f0_hz")]
        profiles[speaker] = C.build_profile(
            speaker,
            gender=_mode(scan.genders),
            age=_mode(scan.ages),
            location=_mode(scan.locations),
            f0_values=f0_values,
            texts=[i["text"] for i in items],
            durations=[i["duration"] for i in items],
            clips=len(items),
        )
    return profiles


def _rows_for(
    ids: list[str],
    clips: dict[str, dict[str, Any]],
    profiles: dict[str, C.SpeakerProfile],
    cfg: PrepareConfig,
    *,
    variants: int,
    dropout: float,
    salt: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audio_id in ids:
        info = clips.get(audio_id)
        if not info:
            continue
        profile = profiles[info["speaker"]]
        transcript = info["text"]
        plain = C.drop_caption(f"{salt}:{audio_id}", dropout)
        captions = [] if plain else C.caption_variants(profile.attrs(), profile.speaker, variants)
        if not captions:
            captions = [""]
        for caption in captions:
            rows.append(
                {
                    "audio": info["audio"],
                    "text": C.caption_with_text(caption, transcript),
                    "duration": info["duration"],
                    "dataset_id": 0,
                    "speaker": info["speaker"],
                    "caption": caption,
                }
            )
    return rows


def is_prepared(paths: Paths, cfg: PrepareConfig) -> bool:
    if not (paths.train_manifest.exists() and paths.data_stats.exists()):
        return False
    try:
        stats = json.loads(paths.data_stats.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    has_val = stats.get("val", {}).get("clips", 0) > 0
    if has_val and not paths.val_manifest.exists():
        return False
    return stats.get("fingerprint") == cfg.fingerprint()


def prepare(paths: Paths, cfg: PrepareConfig | None = None, *, log=print) -> dict[str, Any]:
    """Stream -> 16 kHz WAVs -> caption-conditioned train/val manifests. Idempotent."""
    cfg = cfg or PrepareConfig()
    paths.ensure()

    if is_prepared(paths, cfg) and not cfg.force:
        log(f"[data] already prepared, reusing {paths.manifest_dir}")
        return json.loads(paths.data_stats.read_text(encoding="utf-8"))

    scans, texts, source_rows = scan_metadata(cfg, log=log)
    train_speakers, val_speakers = select_speakers(scans, cfg, log=log)

    train_ids = pick_clips(scans, train_speakers, cfg, cfg.max_clips_per_speaker, salt="train")
    val_ids = pick_clips(scans, val_speakers, cfg, cfg.val_clips_per_speaker, salt="val")
    wanted = {**train_ids, **val_ids}

    clips = fetch_audio(paths, scans, texts, wanted, cfg, log=log)
    for audio_id, info in clips.items():
        info["text"] = texts[audio_id]
    # Drop clips the audio pass could not recover.
    wanted = {s: [i for i in ids if i in clips] for s, ids in wanted.items()}
    train_ids = {s: ids for s, ids in wanted.items() if s in set(train_speakers)}
    val_ids = {s: ids for s, ids in wanted.items() if s in set(val_speakers)}

    profiles = build_profiles(scans, clips, cfg)
    for speaker in val_speakers:
        if speaker in profiles:
            profiles[speaker] = C.SpeakerProfile(**{**profiles[speaker].as_dict(), "val": True})

    if not any(train_ids.values()):
        raise RuntimeError("no training clips after filtering - refusing to write empty manifests")

    train_rows = _rows_for(
        [i for ids in train_ids.values() for i in ids],
        clips,
        profiles,
        cfg,
        variants=cfg.caption_variants,
        dropout=cfg.caption_dropout,
        salt="train",
    )
    val_rows = _rows_for(
        [i for ids in val_ids.values() for i in ids],
        clips,
        profiles,
        cfg,
        variants=1,
        dropout=0.0,
        salt="val",
    )

    n_train = M.write_manifest(paths.train_manifest, train_rows)
    n_val = M.write_manifest(paths.val_manifest, val_rows) if val_rows else 0
    if not val_rows:
        paths.val_manifest.unlink(missing_ok=True)  # upstream treats "" as "no validation"
    paths.speaker_profiles.write_text(
        json.dumps({s: p.as_dict() for s, p in profiles.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths.excluded_speakers.write_text(json.dumps(val_speakers, indent=2), encoding="utf-8")
    log(f"[data] wrote {n_train} train rows / {n_val} val rows")

    train_rep = M.validate_manifest(paths.train_manifest, sample_rate=cfg.target_sr, min_sec=cfg.min_dur, max_sec=cfg.max_dur)
    val_rep = (
        M.validate_manifest(paths.val_manifest, sample_rate=cfg.target_sr, min_sec=cfg.min_dur, max_sec=cfg.max_dur)
        if val_rows
        else M.Report(manifest=paths.val_manifest)
    )

    genders = Counter(p.gender or "unknown" for p in profiles.values() if p.speaker in train_ids)
    stats = {
        "fingerprint": cfg.fingerprint(),
        "dataset": cfg.resolved_dataset,
        "split": cfg.resolved_split,
        "source_rows": source_rows,
        "speakers": {"train": len(train_ids), "val": len(val_ids)},
        "gender": dict(genders),
        "locations": dict(Counter(p.location for p in profiles.values() if p.location).most_common(12)),
        "captions": {
            "variants": cfg.caption_variants,
            "dropout": cfg.caption_dropout,
            "examples": [p.caption for p in list(profiles.values())[:8]],
            "lengths": sorted(len(p.caption) for p in profiles.values()),
        },
        "train": train_rep.summary(),
        "val": val_rep.summary(),
        "f0_hz": {
            "min": min((p.f0_hz for p in profiles.values() if p.f0_hz), default=None),
            "max": max((p.f0_hz for p in profiles.values() if p.f0_hz), default=None),
            "measured_speakers": sum(1 for p in profiles.values() if p.f0_hz),
        },
    }
    paths.data_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    M.write_data_report(
        paths.data_report,
        [train_rep, val_rep],
        extra={
            "dataset": cfg.resolved_dataset,
            "dataset type": "Kenyan Swahili (caption-conditioned voice design)",
            "target sample rate": f"{cfg.target_sr} Hz (AudioVAE encoder input)",
            "speakers": f"{len(train_ids)} train / {len(val_ids)} val (speaker-disjoint)",
            "caption variants per clip": cfg.caption_variants,
            "caption dropout": cfg.caption_dropout,
            "seed": cfg.seed,
        },
    )

    if not train_rep.ok or not val_rep.ok:
        raise RuntimeError(f"manifest validation failed, see {paths.data_report}")

    log(
        f"[data] train {train_rep.total_seconds / 3600:.2f} h / {train_rep.rows} rows | "
        f"val {val_rep.total_seconds / 60:.1f} min / {val_rep.rows} rows"
    )
    return stats

