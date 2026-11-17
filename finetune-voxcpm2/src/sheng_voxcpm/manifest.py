"""JSONL manifest building/validation for VoxCPM2 fine-tuning.

Official format (one JSON object per line):

    {"audio": "wav16k/abel_0000.wav", "text": "...", "duration": 30.0, "dataset_id": 0}

`audio` / `text` are required; `duration`, `dataset_id`, `ref_audio` optional.
Relative `audio` paths are resolved by VoxCPM's loader **against the manifest's
own directory**, which is why we store paths relative to `manifests/`.

Pure stdlib on purpose: this runs on a fresh Colab VM before pip finishes.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

MIN_CLIP_SEC = 1.0
MAX_CLIP_SEC = 30.0


def audio_duration(path: Path) -> float:
    """Duration in seconds. WAV via stdlib, anything else via soundfile."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError):
        pass
    import soundfile as sf  # local import: heavy, only needed for non-WAV

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def wav_sample_rate(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate()
    except (wave.Error, EOFError, OSError):
        import soundfile as sf

        return int(sf.info(str(path)).samplerate)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            row["__line__"] = line_no
            rows.append(row)
    return rows


def write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            row = {k: v for k, v in row.items() if not k.startswith("__")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


@dataclass
class Report:
    manifest: Path
    rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    text_lengths: list[int] = field(default_factory=list)
    speakers: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def total_seconds(self) -> float:
        return float(sum(self.durations))

    def summary(self) -> dict[str, Any]:
        d = sorted(self.durations)
        n = len(d)

        def pct(p: float) -> float:
            return d[min(n - 1, int(p * n))] if n else 0.0

        return {
            "manifest": str(self.manifest),
            "clips": self.rows,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "duration_hours": round(self.total_seconds / 3600, 4),
            "duration_min": round(self.total_seconds / 60, 2),
            "dur_s_mean": round(sum(d) / n, 2) if n else 0,
            "dur_s_min": round(d[0], 2) if n else 0,
            "dur_s_p50": round(pct(0.5), 2),
            "dur_s_max": round(d[-1], 2) if n else 0,
            "text_chars_mean": round(sum(self.text_lengths) / len(self.text_lengths), 1) if self.text_lengths else 0,
            "speakers": self.speakers,
        }

    def render(self) -> str:
        s = self.summary()
        lines = [
            f"### `{self.manifest.name}`",
            "",
            f"- clips: **{s['clips']}**  ({s['duration_min']} min, {s['duration_hours']} h)",
            f"- duration s: min {s['dur_s_min']} / p50 {s['dur_s_p50']} / mean {s['dur_s_mean']} / max {s['dur_s_max']}",
            f"- transcript chars (mean): {s['text_chars_mean']}",
            f"- errors: {len(self.errors)}  warnings: {len(self.warnings)}",
        ]
        if self.speakers:
            lines.append(f"- speakers: {self.speakers}")
        for e in self.errors[:20]:
            lines.append(f"- ERROR: {e}")
        for w in self.warnings[:20]:
            lines.append(f"- warn: {w}")
        return "\n".join(lines)


def validate_manifest(
    path: Path,
    *,
    sample_rate: int = 16_000,
    min_sec: float = MIN_CLIP_SEC,
    max_sec: float = MAX_CLIP_SEC,
    check_sample_rate: bool = True,
) -> Report:
    """Mirror of the checks `voxcpm validate` performs, offline and dependency-light."""
    path = Path(path)
    rep = Report(manifest=path)
    if not path.exists():
        rep.errors.append(f"manifest not found: {path}")
        return rep

    base = path.parent
    rows = read_manifest(path)
    rep.rows = len(rows)
    if not rows:
        rep.errors.append("manifest is empty")

    for row in rows:
        line = row["__line__"]
        audio = row.get("audio")
        text = row.get("text")

        if not audio:
            rep.errors.append(f"line {line}: missing 'audio'")
            continue
        if not isinstance(text, str) or not text.strip():
            rep.errors.append(f"line {line}: missing/empty 'text'")

        wav = Path(audio)
        wav = wav if wav.is_absolute() else (base / wav)
        if not wav.exists():
            rep.errors.append(f"line {line}: audio not found: {wav}")
            continue
        try:
            dur = audio_duration(wav)
            sr = wav_sample_rate(wav)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            rep.errors.append(f"line {line}: cannot read {wav.name}: {exc}")
            continue

        rep.durations.append(dur)
        if isinstance(text, str):
            rep.text_lengths.append(len(text.strip()))
        if dur < min_sec:
            rep.warnings.append(f"line {line}: {dur:.2f}s < {min_sec}s (unstable, will be dropped)")
        elif dur > max_sec:
            rep.warnings.append(
                f"line {line}: {dur:.2f}s > {max_sec}s (VRAM heavy; may be filtered by max_batch_tokens)"
            )
        if check_sample_rate and sr != sample_rate:
            rep.errors.append(
                f"line {line}: sample rate {sr} != {sample_rate} (AudioVAE encoder needs {sample_rate})"
            )
        declared = row.get("duration")
        if declared is not None and abs(float(declared) - dur) > 0.25:
            rep.warnings.append(f"line {line}: declared duration {declared} vs actual {dur:.2f}")

        ref = row.get("ref_audio")
        if ref:
            ref_path = Path(ref)
            ref_path = ref_path if ref_path.is_absolute() else (base / ref_path)
            if not ref_path.exists():
                rep.errors.append(f"line {line}: ref_audio not found: {ref_path}")

        speaker = row.get("speaker")
        if speaker:
            rep.speakers[speaker] = rep.speakers.get(speaker, 0) + 1

    return rep


def write_data_report(path: Path, reports: list[Report], extra: dict[str, Any] | None = None) -> Path:
    lines = ["# Dataset report", ""]
    if extra:
        for k, v in extra.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    for rep in reports:
        lines.append(rep.render())
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
