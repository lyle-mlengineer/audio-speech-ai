from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from sauti_sheng.data import (
    apply_fades,
    choose_reference,
    decode_clip,
    find_silence_cut,
    peak_normalise,
    split_indices,
    write_f5_dataset,
    write_metadata_csv,
)
from sauti_sheng.paths import Paths

SR = 24000


def _bursts(secs: float = 3.0, gap=(1.0, 1.4), amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * secs)) / SR
    x = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    lo, hi = int(gap[0] * SR), int(gap[1] * SR)
    x[lo:hi] = 0.0
    return x


def test_split_indices_is_deterministic_and_disjoint():
    a, b = split_indices(42, 4, 42)
    c, d = split_indices(42, 4, 42)
    assert (a, b) == (c, d)
    assert len(a) + len(b) == 42
    assert set(a).isdisjoint(b)
    assert len(b) == 4
    assert a == sorted(a) and b == sorted(b)


def test_split_indices_edge_cases():
    assert split_indices(0, 4, 1) == ([], [])
    assert split_indices(1, 4, 1) == ([0], [])
    train, val = split_indices(3, 99, 1)
    assert len(val) == 2 and len(train) == 1


def test_find_silence_cut_finds_the_gap():
    found = find_silence_cut(_bursts(), SR, target_seconds=1.2)
    assert found is not None
    cut, silence = found
    assert 1.0 <= cut <= 1.4
    assert silence >= 0.08


def test_find_silence_cut_returns_none_when_far_away():
    assert find_silence_cut(_bursts(), SR, target_seconds=2.8, search_seconds=0.2) is None


def test_peak_normalise_only_scales_loud_clips():
    x = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    assert np.array_equal(peak_normalise(x), x)
    loud = peak_normalise(np.array([0.0, 2.0], dtype=np.float32))
    assert np.max(np.abs(loud)) == pytest.approx(0.99)


def test_apply_fades_touches_only_the_edges():
    x = np.ones(SR // 10, dtype=np.float32)
    y = apply_fades(x, SR, fade_ms=10)
    assert y[0] == 0.0 and y[-1] == 0.0
    assert y[len(y) // 2] == 1.0


def test_decode_clip_reads_bytes_and_downmixes():
    import soundfile as sf

    stereo = np.stack([np.zeros(100, np.float32), np.zeros(100, np.float32)], axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, stereo, 22050, format="WAV")
    audio, sr = decode_clip({"bytes": buffer.getvalue()})
    assert sr == 22050
    assert audio.ndim == 1
    assert audio.dtype == np.float32


def test_decode_clip_rejects_unknown_shape():
    with pytest.raises(ValueError, match="unsupported audio field"):
        decode_clip({"nope": 1})


def test_write_metadata_csv_uses_pipe(tmp_path):
    path = tmp_path / "meta.csv"
    write_metadata_csv(
        [{"audio_path": "/a.wav", "text": "Niaje mazee"}, {"audio_path": "/b.wav", "text": "Sasa"}],
        path,
    )
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    assert rows[0] == {"audio_file": "/a.wav", "text": "Niaje mazee"}
    assert len(rows) == 2


def test_write_f5_dataset_roundtrips(tmp_path):
    paths = Paths(tmp_path / "artifacts").ensure()
    records = [
        {"audio_path": str(tmp_path / "a.wav"), "text": "Niaje mazee.", "duration": 1.5},
        {"audio_path": str(tmp_path / "b.wav"), "text": "Sasa.", "duration": 1.0},
    ]
    write_f5_dataset(records, [1.5, 1.0], paths)

    from datasets import Dataset

    data = Dataset.from_file(str(paths.f5_raw_arrow))
    assert data.column_names == ["audio_path", "text", "duration"]
    assert data[0]["text"] == "Niaje mazee."
    durations = json.loads(paths.f5_duration_json.read_text(encoding="utf-8"))
    assert durations == {"duration": [1.5, 1.0]}


def test_choose_reference_prefers_a_real_pause():
    text = "Moja ni hii. Mbili ni hiyo. Tatu ni ile. Nne ni zile. Tano ni hizo."
    candidates = [
        {"index": 0, "utt_id": "utt_0001", "audio": _bursts(3.0), "sr": SR, "text": text, "duration": 3.0},
        {"index": 1, "utt_id": "utt_0002", "audio": _bursts(3.0, amp=0.2), "sr": SR, "text": text, "duration": 3.0},
    ]
    choice, audio = choose_reference(candidates, target_seconds=1.2)
    assert choice.confidence in {"silence", "short-silence"}
    assert choice.cut_seconds > 0
    assert len(audio) == int(choice.cut_seconds * SR)
    assert choice.text
