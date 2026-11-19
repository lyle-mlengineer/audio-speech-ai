"""End-to-end data stage on a synthetic parquet (no network, no GPU)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from sheng_qwentts import data
from sheng_qwentts.paths import Paths


def _wav_bytes(freq: float = 220.0, seconds: float = 2.0, sr: int = 22050, gain: float = 0.3) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    wav = (gain * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = sf.write
    import io

    bio = io.BytesIO()
    sf.write(bio, wav, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


def _write_parquet(path, n: int = 12):
    rows = [
        {
            "audio": {"bytes": _wav_bytes(freq=200 + 10 * i), "path": f"clip_{i}.wav"},
            "speaker": "abel",
            "text": f"Mambo vipi mazee, hii ni stori ya nambari {i} kwa Sheng.",
        }
        for i in range(n)
    ]
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                ("speaker", pa.string()),
                ("text", pa.string()),
            ]
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


@pytest.fixture()
def fake_parquet(tmp_path, monkeypatch) -> Paths:
    paths = Paths(tmp_path / "artifacts").ensure()
    parquet = tmp_path / "fake.parquet"
    _write_parquet(parquet)
    monkeypatch.setattr(data, "download_parquet", lambda paths, force=False: parquet)
    return paths


def test_build_writes_24k_wavs_and_manifests(fake_parquet):
    paths = fake_parquet
    info = data.build(paths, val_size=3, seed=7)

    assert info["clips"] == 12
    assert info["train_clips"] == 9 and info["val_clips"] == 3
    assert info["sample_rate"] == 24000

    wavs = sorted(paths.wav_dir.glob("*.wav"))
    assert len(wavs) == 12
    for wav in wavs[:3]:
        got = sf.info(wav)
        assert got.samplerate == 24000
        assert got.channels == 1
        assert got.duration == pytest.approx(2.0, abs=0.01)

    train = [json.loads(line) for line in paths.train_raw_jsonl.open(encoding="utf-8")]
    val = [json.loads(line) for line in paths.val_raw_jsonl.open(encoding="utf-8")]
    assert len(train) == 9 and len(val) == 3
    assert {"audio", "text", "ref_audio", "duration"} <= set(train[0])
    ref_audio = paths.ref_audio
    assert ref_audio.is_file()
    assert {row["ref_audio"] for row in train + val} == {str(ref_audio)}
    assert sf.info(ref_audio).samplerate == 24000
    assert (paths.data_dir / "ref_text.txt").read_text(encoding="utf-8").strip()


def test_split_is_deterministic(fake_parquet, tmp_path):
    paths = fake_parquet
    first = data.build(paths, val_size=3, seed=11)
    second = data.build(paths, val_size=3, seed=11)
    assert first["val_indices"] == second["val_indices"]
    other = data.build(paths, val_size=3, seed=12)
    assert other["val_indices"] != first["val_indices"] or other["ref_clip"] == first["ref_clip"]


def test_ref_index_is_honoured(fake_parquet):
    paths = fake_parquet
    info = data.build(paths, val_size=3, seed=7, ref_index=2)
    assert info["ref_clip"] == "utt_0002.wav"


def test_ref_index_must_be_a_training_clip(fake_parquet):
    paths = fake_parquet
    data.build(paths, val_size=3, seed=7)
    val_indices = json.loads(paths.dataset_info.read_text(encoding="utf-8"))["val_indices"]
    if not val_indices:
        pytest.skip("no val split")
    with pytest.raises(ValueError, match="not a training clip"):
        data.build(paths, val_size=3, seed=7, ref_index=val_indices[0])


def test_report_mentions_the_numbers(fake_parquet):
    paths = fake_parquet
    info = data.build(paths, val_size=3)
    report = paths.data_report.read_text(encoding="utf-8")
    assert str(info["clips"]) in report
    assert "24 kHz" in report or str(info["sample_rate"]) in report
    assert info["ref_clip"] in report
    assert report.count("| train |") + report.count("| val |") == info["clips"]
