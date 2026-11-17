"""Dataset -> 16 kHz wavs + manifests, exercised with a fake HF dataset.

The audio codecs (torchaudio/soundfile) and the `datasets` loader are replaced, so
this runs on a bare Python: it tests the split, the manifest contract, the stats
schema and idempotency, not resampling quality.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from sheng_voxcpm import dataset as D
from sheng_voxcpm import manifest as M


def _rows(n: int = 8, speaker: str = "abel"):
    return [
        {
            "audio": {"array": [0.0, 0.1, -0.1], "sampling_rate": 22_050},
            "speaker": speaker,
            "text": f"  Niaje msee   number {i}\n",
        }
        for i in range(n)
    ]


class FakeDataset(list):
    """Stand-in for a HF `datasets.Dataset` split."""

    column_names = ["audio", "speaker", "text"]


@pytest.fixture()
def fake_audio(monkeypatch):
    """Fake `datasets.load_dataset` + fake audio I/O + fake WAV metadata."""
    holder: dict = {"rows": FakeDataset(_rows())}
    module = types.ModuleType("datasets")

    def load_dataset(dataset_id, split=None):
        holder["dataset_id"] = dataset_id
        holder["split"] = split
        return holder["rows"]

    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)

    monkeypatch.setattr(D, "_resample", lambda array, src_sr, dst_sr: array)

    def fake_write(path, array, sample_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        return 3.0

    monkeypatch.setattr(D, "_write_wav", fake_write)
    monkeypatch.setattr(M, "audio_duration", lambda path: 3.0)
    monkeypatch.setattr(M, "wav_sample_rate", lambda path: 16_000)
    return holder


def test_prepare_writes_manifests_and_stats(paths, fake_audio):
    stats = D.prepare(paths, D.PrepareConfig(), log=lambda *_: None)

    assert fake_audio["dataset_id"] == D.DATASET_ID
    assert stats["train"]["clips"] + stats["val"]["clips"] == 8
    assert stats["val"]["clips"] == 1  # max(1, min(4, 8 // 5))
    assert stats["target_sr"] == 16_000
    assert stats["source_sample_rates"] == [22_050]

    train = M.read_manifest(paths.train_manifest)
    val = M.read_manifest(paths.val_manifest)
    assert len(train) == 7 and len(val) == 1

    row = train[0]
    assert row["audio"].startswith("..") and row["audio"].endswith(".wav")
    assert row["text"].startswith("Niaje msee number ")  # whitespace collapsed, no newline
    assert "\n" not in row["text"]
    assert row["dataset_id"] == 0
    assert row["duration"] == 3.0
    assert (paths.manifest_dir / row["audio"]).resolve().exists()

    # one file per clip, speaker-prefixed and zero-padded
    wavs = sorted(p.name for p in paths.wav_dir.glob("*.wav"))
    assert wavs[0].startswith("abel_0000") and len(wavs) == 8

    assert stats["code_switch"]["tokens"] > 0
    assert stats["code_switch"]["unique_tokens"] > 2
    report = paths.data_report.read_text(encoding="utf-8")
    assert "Sheng (Swahili + English code-switching)" in report
    assert json.loads(paths.data_stats.read_text(encoding="utf-8"))["dataset_id"] == D.DATASET_ID


def test_prepare_is_idempotent(paths, fake_audio):
    calls = {"n": 0}
    original = sys.modules["datasets"].load_dataset

    def counting(dataset_id, split=None):
        calls["n"] += 1
        return original(dataset_id, split)

    sys.modules["datasets"].load_dataset = counting
    D.prepare(paths, D.PrepareConfig(), log=lambda *_: None)
    D.prepare(paths, D.PrepareConfig(), log=lambda *_: None)
    assert calls["n"] == 1  # second call reuses the manifests


def test_force_rebuilds(paths, fake_audio):
    calls = {"n": 0}
    original = sys.modules["datasets"].load_dataset

    def counting(dataset_id, split=None):
        calls["n"] += 1
        return original(dataset_id, split)

    sys.modules["datasets"].load_dataset = counting
    D.prepare(paths, D.PrepareConfig(), log=lambda *_: None)
    D.prepare(paths, D.PrepareConfig(force=True), log=lambda *_: None)
    assert calls["n"] == 2


def test_speaker_filter(paths, fake_audio):
    fake_audio["rows"] = FakeDataset(_rows(4, speaker="other") + _rows(4))
    stats = D.prepare(paths, D.PrepareConfig(speaker="abel"), log=lambda *_: None)
    assert stats["train"]["clips"] + stats["val"]["clips"] == 4
    assert stats["train"]["speakers"] == {"abel": 3}
    assert any("filtered out" in d for d in stats["drops"])


def test_tiny_dataset_writes_no_val_manifest(paths, fake_audio):
    """One clip cannot be split: train only, and the config must say so."""
    from sheng_voxcpm.config import TrainConfig

    fake_audio["rows"] = FakeDataset(_rows(1))
    stats = D.prepare(paths, D.PrepareConfig(), log=lambda *_: None)
    assert stats["train"]["clips"] == 1
    assert stats["val"]["clips"] == 0
    assert not paths.val_manifest.exists()
    assert TrainConfig().build_payload(paths)["val_manifest"] == ""
    assert D.is_prepared(paths, D.PrepareConfig())  # ...and it stays prepared


def test_val_fraction_and_manifest_path(paths, fake_audio):
    from sheng_voxcpm.config import TrainConfig

    stats = D.prepare(paths, D.PrepareConfig(), log=lambda *_: None)
    assert stats["val"]["clips"] == 1  # ~15% of 8, never more than --val-clips
    assert TrainConfig().build_payload(paths)["val_manifest"] == str(paths.val_manifest)


def test_empty_transcripts_are_dropped(paths, fake_audio):
    fake_audio["rows"][0]["text"] = "   "
    stats = D.prepare(paths, D.PrepareConfig(), log=lambda *_: None)
    assert stats["train"]["clips"] + stats["val"]["clips"] == 7
    assert any("empty transcript" in d for d in stats["drops"])


def test_too_short_clips_are_dropped(paths, fake_audio, monkeypatch):
    monkeypatch.setattr(D, "_write_wav", lambda path, array, sr: 0.4)
    with pytest.raises(RuntimeError, match="no usable clips"):
        D.prepare(paths, D.PrepareConfig(), log=lambda *_: None)


def test_ref_audio_fraction_adds_references(paths, fake_audio):
    stats = D.prepare(paths, D.PrepareConfig(ref_audio_fraction=0.5), log=lambda *_: None)
    train = M.read_manifest(paths.train_manifest)
    with_ref = [r for r in train if "ref_audio" in r]
    assert with_ref, "expected some rows to carry ref_audio"
    assert len(with_ref) == max(1, round(0.5 * len(train)))
    for row in with_ref:
        assert row["ref_audio"] != row["audio"]
        assert row["ref_duration"] == 3.0
        assert (paths.manifest_dir / row["ref_audio"]).resolve().exists()
    assert stats["ref_audio_fraction"] == 0.5


def test_clean_text():
    assert D.clean_text("  a\\u00a0b\\n c  ") == "a b c" or D.clean_text("  a\u00a0b\n c  ") == "a b c"
    assert D.clean_text(None) == ""
    assert D.clean_text("   ") == ""


def test_slug():
    assert D._slug("Abel Mwangi") == "abel-mwangi"
    assert D._slug("") == "speaker"


def test_is_prepared_detects_changed_config(paths, fake_audio):
    D.prepare(paths, D.PrepareConfig(), log=lambda *_: None)
    assert D.is_prepared(paths, D.PrepareConfig())
    assert not D.is_prepared(paths, D.PrepareConfig(val_clip_count=2))
    assert not D.is_prepared(paths, D.PrepareConfig(dataset_id="other/dataset"))


def test_code_switch_stats():
    stats = D.code_switch_stats(["Niaje msee, leo tuna chill tu.", "Habari yako"])
    assert stats["tokens"] == 8
    assert stats["english_marker_share"] > 0  # "chill"
    assert stats["swahili_marker_share"] > 0  # "leo", "tu", "habari", ...
    assert 0 < stats["english_marker_share"] < 1
