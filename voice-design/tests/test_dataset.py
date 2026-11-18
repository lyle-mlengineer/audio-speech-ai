"""Data build: selection, caption wiring and the idempotency that saves a disconnect."""

from __future__ import annotations

import json

import pytest

from kenya_voices import captions as C
from kenya_voices import dataset as D

from conftest import write_wav


def fake_source(n_speakers: int = 8, clips: int = 8) -> tuple[dict[str, D.SpeakerScan], dict[str, str]]:
    scans: dict[str, D.SpeakerScan] = {}
    texts: dict[str, str] = {}
    for s in range(n_speakers):
        speaker = f"spk{s:02d}"
        scan = D.SpeakerScan(speaker=speaker)
        for c in range(clips):
            audio_id = f"{speaker}-{c:03d}"
            scan.clips.append((audio_id, 5.0))
            scan.genders.append("Female" if c % 2 else "Male")
            scan.ages.append("25-35" if s % 2 else "36-49")
            scan.locations.append("Nairobi")
            texts[audio_id] = "Habari yako leo rafiki"
        scans[speaker] = scan
    return scans, texts


class FakeFetch:
    def __init__(self, texts):
        self.texts = texts
        self.calls = 0
        self.ids: list[str] = []

    def __call__(self, paths, scans, texts, wanted, cfg, log=print):
        self.calls += 1
        out = {}
        for speaker, ids in wanted.items():
            gender = "Female" if (int(speaker[-2:]) % 2) else "Male"
            for i, audio_id in enumerate(ids):
                wav = write_wav(paths.wav_dir / f"{speaker}_{i:03d}.wav", seconds=5.0)
                out[audio_id] = {
                    "audio": paths.rel_to_manifest(wav),
                    "duration": 5.0,
                    "speaker": speaker,
                    "f0_hz": 210.0 if gender == "Female" else 120.0,
                }
                self.ids.append(audio_id)
        return out


@pytest.fixture
def wired(paths, monkeypatch):
    scans, texts = fake_source()
    fetch = FakeFetch(texts)
    monkeypatch.setattr(D, "scan_metadata", lambda cfg, log=print: (scans, texts, 64))
    monkeypatch.setattr(D, "fetch_audio", fetch)
    return paths, fetch, scans, texts


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def small_cfg(**kwargs) -> D.PrepareConfig:
    base = dict(
        preset="badrex",
        max_speakers=6,
        max_clips_per_speaker=5,
        min_clips_per_speaker=4,
        max_audio_hours=100.0,
        val_speaker_fraction=0.34,
        val_clips_per_speaker=2,
        caption_dropout=0.0,
        caption_variants=1,
    )
    base.update(kwargs)
    return D.PrepareConfig(**base)


def test_prepare_writes_caption_conditioned_manifests(wired):
    paths, fetch, scans, _ = wired
    stats = D.prepare(paths, small_cfg())
    train, val = rows(paths.train_manifest), rows(paths.val_manifest)
    assert train and val
    assert all(r["text"].startswith("(") for r in train)
    assert all("Habari yako leo rafiki" in r["text"] for r in train)
    assert stats["speakers"]["train"] + stats["speakers"]["val"] == 6
    # Validation speakers are held out entirely.
    assert not {r["speaker"] for r in train} & {r["speaker"] for r in val}
    assert stats["train"]["caption_share"] == 1.0
    assert all(r["dataset_id"] == 0 for r in train)


def test_val_is_capped_and_smaller(wired):
    paths, _, _, _ = wired
    D.prepare(paths, small_cfg())
    train, val = rows(paths.train_manifest), rows(paths.val_manifest)
    per_speaker = {}
    for r in val:
        per_speaker[r["speaker"]] = per_speaker.get(r["speaker"], 0) + 1
    assert all(n <= 2 for n in per_speaker.values())
    assert len(val) < len(train)


def test_caption_dropout_keeps_plain_rows(wired):
    paths, _, _, _ = wired
    D.prepare(paths, small_cfg(caption_dropout=1.0))
    assert all(not r["text"].startswith("(") for r in rows(paths.train_manifest))


def test_caption_variants_multiply_rows(wired):
    paths, _, _, _ = wired
    D.prepare(paths, small_cfg(caption_variants=1))
    one = len(rows(paths.train_manifest))
    D.prepare(paths, small_cfg(caption_variants=2, force=True))
    two = len(rows(paths.train_manifest))
    assert two == pytest.approx(2 * one, abs=2)
    texts = [C.split_caption(r["text"])[0] for r in rows(paths.train_manifest)[:4]]
    assert len(set(texts)) >= 2


def test_prepare_is_idempotent(wired):
    paths, fetch, _, _ = wired
    stats = D.prepare(paths, small_cfg())
    calls = fetch.calls
    again = D.prepare(paths, small_cfg())
    assert fetch.calls == calls  # no second stream
    assert again["fingerprint"] == stats["fingerprint"]


def test_force_rebuilds(wired):
    paths, fetch, _, _ = wired
    D.prepare(paths, small_cfg())
    calls = fetch.calls
    D.prepare(paths, small_cfg(force=True))
    assert fetch.calls == calls + 1


def test_is_prepared_detects_changed_config(wired):
    paths, _, _, _ = wired
    cfg = small_cfg()
    D.prepare(paths, cfg)
    assert D.is_prepared(paths, cfg)
    assert not D.is_prepared(paths, small_cfg(max_speakers=7))
    assert not D.is_prepared(paths, small_cfg(caption_dropout=0.2))


def test_profiles_and_stats_files(wired):
    paths, _, _, _ = wired
    stats = D.prepare(paths, small_cfg())
    profiles = json.loads(paths.speaker_profiles.read_text(encoding="utf-8"))
    assert profiles and all(p["caption"] for p in profiles.values())
    assert json.loads(paths.excluded_speakers.read_text(encoding="utf-8"))
    # val speakers only have 2 clips each -> too few f0 measurements to label pitch
    assert stats["f0_hz"]["measured_speakers"] >= 4
    assert "Nairobi" in stats["locations"]


# ---------------------------------------------------------------- pure logic
def test_select_speakers_is_deterministic_and_disjoint():
    scans, _ = fake_source(12, 8)
    cfg = D.PrepareConfig(max_speakers=10, min_clips_per_speaker=4, seed=7, val_speaker_fraction=0.2)
    train_a, val_a = D.select_speakers(scans, cfg)
    train_b, val_b = D.select_speakers(scans, cfg)
    assert (train_a, val_a) == (train_b, val_b)
    assert not set(train_a) & set(val_a)
    assert 1 <= len(val_a) <= 3
    assert len(train_a) + len(val_a) <= 10


def test_select_speakers_respects_audio_budget():
    scans, _ = fake_source(20, 8)  # 40 s per speaker
    cfg = D.PrepareConfig(max_speakers=50, min_clips_per_speaker=4, max_audio_hours=0.05)  # 180 s
    train, val = D.select_speakers(scans, cfg)
    assert 2 <= len(train) + len(val) <= 6


def test_select_speakers_rejects_tiny_speakers():
    scans, _ = fake_source(3, 2)
    with pytest.raises(RuntimeError, match="min-clips-per-speaker"):
        D.select_speakers(scans, D.PrepareConfig(min_clips_per_speaker=6))


def test_pick_clips_is_capped_and_stable():
    scans, _ = fake_source(4, 10)
    cfg = D.PrepareConfig(seed=3)
    first = D.pick_clips(scans, ["spk00", "spk01"], cfg, 4, salt="train")
    second = D.pick_clips(scans, ["spk00", "spk01"], cfg, 4, salt="train")
    assert first == second
    assert all(len(ids) == 4 for ids in first.values())
    assert len(set(D.pick_clips(scans, ["spk00"], cfg, 4, "val")["spk00"])) == 4


def test_rows_for_skips_missing_audio():
    scans, texts = fake_source(2, 3)
    clips = {
        "spk00-000": {"audio": "wav16k/a.wav", "duration": 5.0, "speaker": "spk00", "f0_hz": 120.0, "text": texts["spk00-000"]},
    }
    profiles = D.build_profiles(scans, clips, D.PrepareConfig())
    out = D._rows_for(["spk00-000", "spk00-001"], clips, profiles, D.PrepareConfig(), variants=1, dropout=0.0, salt="t")
    assert len(out) == 1
    assert out[0]["audio"] == "wav16k/a.wav"
    assert out[0]["text"].startswith("(")


def test_clean_text():
    assert D.clean_text("  Habari\n yako\u00a0leo ") == "Habari yako leo"
    assert D.clean_text("") == ""


def test_stream_columns_are_metadata_only():
    # Guard: the badrex metadata pass must never request the audio column.
    cols = D.PrepareConfig().meta_columns()
    assert "audio" not in cols
    assert "transcription" in cols


def test_preset_without_metadata_uses_generic_columns():
    cfg = D.PrepareConfig(preset="kenyan-voices")
    assert cfg.columns()["text"] == "text"
    assert "audio" in cfg.meta_columns()  # duration has to be measured


class FakeStream:
    """Minimal stand-in for a streamed HF dataset (supports filter(input_columns))."""

    def __init__(self, rows):
        self.rows = rows

    def filter(self, fn, input_columns=None):
        return FakeStream([row for row in self.rows if fn({key: row[key] for key in input_columns})])

    def __iter__(self):
        return iter(self.rows)


def test_scan_and_fetch_against_a_streamed_dataset(paths, monkeypatch):
    rows = []
    for s in range(3):
        for c in range(8):
            rows.append(
                {
                    "audio_id": f"a{s}{c}",
                    "voice_creator_id": f"spk{s}",
                    "audio_duration": 6.0,
                    "gender": "Female" if s else "Male",
                    "age_group": "25-35",
                    "location": "Nairobi",
                    "transcription": "Habari yako leo rafiki",
                    "audio": {"array": [0.1, -0.1] * 48_000, "sampling_rate": 16_000},
                }
            )
    monkeypatch.setattr(D, "_stream", lambda *a, **k: FakeStream(rows))
    monkeypatch.setattr(D, "_write_wav", lambda path, array, sr: 3.0)
    monkeypatch.setattr(D, "median_f0", lambda *a, **k: 120.0)

    cfg = D.PrepareConfig(max_speakers=3, min_clips_per_speaker=4, max_audio_hours=100.0)
    scans, texts, scanned = D.scan_metadata(cfg)
    assert scanned == 24 and len(scans) == 3
    assert scans["spk1"].genders[:2] == ["Female", "Female"]

    wanted = D.pick_clips(scans, list(scans), cfg, 3, salt="train")
    clips = D.fetch_audio(paths, scans, texts, wanted, cfg)
    assert len(clips) == 9
    assert all(c["text"] == "Habari yako leo rafiki" for c in clips.values())
    assert all(c["f0_hz"] == 120.0 for c in clips.values())


def test_wav_names_depend_only_on_speaker_and_clip_id(paths):
    a = D.wav_path_for(paths, "spk01", "abc123")
    assert a == D.wav_path_for(paths, "spk01", "abc123")
    assert a != D.wav_path_for(paths, "spk02", "abc123")
    assert a.parent == paths.wav_dir
    assert len(a.name) < 60
