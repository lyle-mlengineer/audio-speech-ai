"""Manifest contract: the exact JSONL VoxCPM2's trainer reads."""

from __future__ import annotations

import json

from kenya_voices import manifest as M


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_write_and_read_roundtrip(tmp_path, manifest_row):
    row = manifest_row()
    path = tmp_path / "train.jsonl"
    assert M.write_manifest(path, [row, row]) == 2
    rows = read_jsonl(path)
    assert rows[0]["audio"] == row["audio"]
    assert rows[0]["text"] == row["text"]
    assert "__line__" not in rows[0]
    assert [r["__line__"] for r in M.read_manifest(path)] == [1, 2]


def test_validation_accepts_a_good_manifest(tmp_path, manifest_row):
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [manifest_row("a.wav"), manifest_row("b.wav")])
    rep = M.validate_manifest(path)
    assert rep.ok, rep.errors
    assert rep.rows == 2
    assert rep.caption_lengths and rep.summary()["caption_share"] == 1.0


def test_caption_without_transcript_is_an_error(tmp_path, manifest_row):
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [manifest_row(text="(A Kenyan male voice)")])
    rep = M.validate_manifest(path)
    assert not rep.ok
    assert any("caption without transcript" in e for e in rep.errors)


def test_long_caption_is_a_warning(tmp_path, manifest_row):
    caption = "A very " + "very " * 40 + "long Kenyan voice"
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [manifest_row(text=f"({caption})Habari")])
    rep = M.validate_manifest(path)
    assert rep.ok
    assert any("dilutes the text" in w for w in rep.warnings)


def test_missing_audio_file_is_an_error(tmp_path, manifest_row):
    row = manifest_row()
    row["audio"] = "wav16k/nope.wav"
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [row])
    rep = M.validate_manifest(path)
    assert not rep.ok
    assert any("audio not found" in e for e in rep.errors)


def test_wrong_sample_rate_is_an_error(tmp_path, wav_factory):
    wav = wav_factory("bad.wav", sample_rate=22_050)
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [{"audio": wav.name, "text": "Habari yako", "duration": 2.0}])
    rep = M.validate_manifest(path, sample_rate=16_000)
    assert not rep.ok
    assert any("sample rate" in e for e in rep.errors)


def test_missing_fields_and_short_clips(tmp_path, wav_factory):
    wav = wav_factory("short.wav", seconds=0.5)
    path = tmp_path / "train.jsonl"
    M.write_manifest(
        path,
        [
            {"text": "Habari"},
            {"audio": wav.name, "text": "  "},
            {"audio": wav.name, "text": "Habari", "duration": 0.5},
        ],
    )
    rep = M.validate_manifest(path, min_sec=1.0)
    assert any("missing 'audio'" in e for e in rep.errors)
    assert any("missing/empty 'text'" in e for e in rep.errors)
    assert any("< 1.0" in w for w in rep.warnings)


def test_declared_duration_mismatch_warns(tmp_path, manifest_row):
    row = manifest_row()
    row["duration"] = 9.0
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [row])
    rep = M.validate_manifest(path)
    assert rep.ok
    assert any("declared duration" in w for w in rep.warnings)


def test_missing_ref_audio_is_an_error(tmp_path, manifest_row):
    row = manifest_row()
    row["ref_audio"] = "nope.wav"
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [row])
    assert any("ref_audio not found" in e for e in M.validate_manifest(path).errors)


def test_report_summary_speakers_and_durations(tmp_path, manifest_row):
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [manifest_row("a.wav", seconds=2.0), manifest_row("b.wav", seconds=4.0)])
    summary = M.validate_manifest(path).summary()
    assert summary["clips"] == 2
    assert summary["duration_min"] == round(6 / 60, 2)
    assert summary["dur_s_max"] == 4.0
    assert summary["speakers"] == {"spk": 2}


def test_write_data_report(tmp_path, manifest_row):
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [manifest_row()])
    report = M.write_data_report(tmp_path / "report.md", [M.validate_manifest(path)], extra={"dataset": "x"})
    text = report.read_text(encoding="utf-8")
    assert "Dataset report" in text
    assert "dataset: x" in text
    assert "caption share" in text


def test_empty_manifest_is_an_error(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("", encoding="utf-8")
    rep = M.validate_manifest(path)
    assert not rep.ok
    assert any("empty" in e for e in rep.errors)


def test_read_manifest_rejects_bad_json(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    try:
        M.read_manifest(path)
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_json_is_utf8_and_one_row_per_line(tmp_path, manifest_row):
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [manifest_row(text="(A Kenyan voice)Habari yako, asante sana")])
    raw = path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 1
    assert json.loads(raw[0])["text"].startswith("(A Kenyan voice)")
