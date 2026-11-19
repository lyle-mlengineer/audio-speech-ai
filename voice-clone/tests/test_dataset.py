"""The data stage: parquet/local dir in, 16 kHz wavs + clean splits out."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from sheng_voiceclone import dataset
from sheng_voiceclone import manifest as mf
from sheng_voiceclone.paths import SAMPLE_RATE, SOURCE_SAMPLE_RATE


def _wav_bytes(seconds: float, sr: int = SOURCE_SAMPLE_RATE) -> bytes:
    t = np.arange(int(seconds * sr), dtype=np.float32) / float(sr)
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def make_parquet(tmp_path, n=8, seconds=1.2):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "audio": {"bytes": _wav_bytes(seconds), "path": f"u{i}.wav"},
            "text": f"sasa bro uko aje {i}",
            "speaker": "abel",
        }
        for i in range(n)
    ]
    p = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pylist(rows), p)
    return p


def test_read_clips_resamples_and_writes_16k(tmp_path):
    pq_path = make_parquet(tmp_path)
    clips = dataset.read_clips(pq_path, tmp_path / "wavs")
    assert len(clips) == 8
    for c in clips:
        assert c.wav.is_file()
        assert sf.info(str(c.wav)).samplerate == SAMPLE_RATE
        assert sf.info(str(c.wav)).channels == 1
        assert c.seconds == pytest.approx(1.2, abs=0.02)


def test_read_clips_is_idempotent(tmp_path):
    pq_path = make_parquet(tmp_path)
    first = dataset.read_clips(pq_path, tmp_path / "wavs")
    mtimes = {c.wav: c.wav.stat().st_mtime_ns for c in first}
    second = dataset.read_clips(pq_path, tmp_path / "wavs")
    assert {c.wav: c.wav.stat().st_mtime_ns for c in second} == mtimes


def test_build_writes_splits_reference_and_report(art, tmp_path):
    pq_path = make_parquet(tmp_path)
    info = dataset.build(art, parquet=pq_path, val_clips=2, test_clips=2, seed=1)

    assert info["clips"] == 8
    assert (info["train"], info["val"], info["test"]) == (4, 2, 2)
    assert art.ref_audio.is_file() and art.ref_text.is_file()
    assert (art.data_dir / "data_report.md").is_file()
    assert json.loads((art.data_dir / "data_info.json").read_text())["sample_rate"] == SAMPLE_RATE

    train = mf.load(art.train_manifest, root=art.root)
    val = mf.load(art.val_manifest, root=art.root)
    test = mf.load(art.test_manifest, root=art.root)
    assert (len(train), len(val), len(test)) == (4, 2, 2)


def test_reference_clip_is_held_out_of_training(art, tmp_path):
    pq_path = make_parquet(tmp_path)
    dataset.build(art, parquet=pq_path, val_clips=2, test_clips=2, seed=1)
    train = mf.load(art.train_manifest, root=art.root)
    ref_text = art.ref_text.read_text(encoding="utf-8").strip()
    assert ref_text not in {r["text"] for r in train}


def test_reference_is_the_longest_validation_clip(art, tmp_path):
    pq_path = make_parquet(tmp_path)
    info = dataset.build(art, parquet=pq_path, val_clips=2, test_clips=2, seed=3)
    val = mf.load(art.val_manifest, root=art.root)
    assert info["ref_seconds"] == pytest.approx(max(r["seconds"] for r in val), abs=0.02)


def test_manifest_rows_are_relative_and_complete(art, tmp_path):
    pq_path = make_parquet(tmp_path)
    dataset.build(art, parquet=pq_path, val_clips=2, test_clips=2, seed=1)
    row = mf.load(art.train_manifest, root=art.root)[0]
    assert not row["audio"].startswith("/") and not row["audio"][1:3] == ":/"
    assert row["language"] == "sw" and row["speaker"] == "abel"
    assert row["seconds"] > 0


def test_split_is_reproducible_and_disjoint():
    from sheng_voiceclone.dataset import Clip

    clips = [Clip(index=i, text=f"t{i}", speaker="abel", seconds=1.0, wav=None) for i in range(1, 9)]
    a = dataset.split(clips, val_clips=2, test_clips=2, seed=7)
    b = dataset.split(clips, val_clips=2, test_clips=2, seed=7)
    assert [[c.index for c in part] for part in a] == [[c.index for c in part] for part in b]
    train, val, test = a
    assert not ({c.index for c in train} & {c.index for c in val})
    assert not ({c.index for c in test} & {c.index for c in val})


def test_too_few_clips_is_rejected():
    from sheng_voiceclone.dataset import Clip

    clips = [Clip(index=i, text="t", speaker="abel", seconds=1.0, wav=None) for i in range(5)]
    with pytest.raises(RuntimeError, match="need >="):
        dataset.split(clips, val_clips=2, test_clips=2)


def test_read_dir_uses_sidecar_transcripts(tmp_path):
    src = tmp_path / "mine"
    src.mkdir()
    for i in range(3):
        sf.write(str(src / f"rec{i}.wav"), np.zeros(16000, dtype=np.float32), 16000)
        (src / f"rec{i}.txt").write_text(f"sentensi ya {i}", encoding="utf-8")
    clips = dataset.read_dir(src, tmp_path / "wavs")
    assert [c.text for c in clips] == ["sentensi ya 0", "sentensi ya 1", "sentensi ya 2"]


def test_read_dir_requires_a_transcript(tmp_path):
    src = tmp_path / "mine"
    src.mkdir()
    sf.write(str(src / "no_text.wav"), np.zeros(16000, dtype=np.float32), 16000)
    with pytest.raises(RuntimeError, match="no transcript"):
        dataset.read_dir(src, tmp_path / "wavs")


def test_read_dir_accepts_transcripts_csv(tmp_path):
    src = tmp_path / "mine"
    src.mkdir()
    sf.write(str(src / "a.wav"), np.zeros(16000, dtype=np.float32), 16000)
    (src / "transcripts.csv").write_text("file,text\na.wav,niaje wewe\n", encoding="utf-8")
    assert dataset.read_dir(src, tmp_path / "wavs")[0].text == "niaje wewe"
