"""Shared fixtures. Tests import the package from ``src/`` (no install needed).

No GPU, no torch, no ``datasets`` and no ``soundfile``: everything here runs on a
bare Python so the pipeline logic can be checked before a Colab session starts.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kenya_voices.paths import Paths  # noqa: E402 - needs sys.path above


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    p = Paths(artifacts=tmp_path / "artifacts", code_root=tmp_path / "repo")
    p.ensure()
    return p


def write_wav(path: Path, seconds: float = 2.0, f0: float = 180.0, sample_rate: int = 16_000) -> Path:
    """16 kHz mono PCM16 sine - good enough for duration / sample-rate checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        samples = (
            struct.pack("<h", int(12_000 * math.sin(2 * math.pi * f0 * i / sample_rate))) for i in range(frames)
        )
        w.writeframes(b"".join(samples))
    return path


@pytest.fixture
def wav_factory(tmp_path: Path):
    def _make(name: str = "clip.wav", **kwargs) -> Path:
        return write_wav(tmp_path / name, **kwargs)

    return _make


@pytest.fixture
def manifest_row(wav_factory, tmp_path: Path):
    def _row(name: str = "a.wav", text: str = "(A Kenyan male voice)Habari yako", **kwargs) -> dict:
        wav = wav_factory(name, **kwargs)
        return {
            "audio": str(wav.relative_to(tmp_path)),
            "text": text,
            "duration": round(kwargs.get("seconds", 2.0), 3),
            "dataset_id": 0,
            "speaker": "spk",
        }

    return _row


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
