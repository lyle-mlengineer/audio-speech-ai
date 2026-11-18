"""Manifest building/validation must match what ``train_t3`` reads."""

from __future__ import annotations

from sheng_chatterbox import manifest as M


def test_write_and_read_roundtrip(tmp_path, wav_factory):
    wav = wav_factory(tmp_path / "wav16k" / "abel_0000.wav", seconds=4.0)
    rows = [
        {
            "audio": wav.name,
            "text": "Niaje msee, leo tuna chill.",
            "duration": 4.0,
            "speaker": "abel",
            "language": "sw",
        }
    ]
    path = tmp_path / "manifests" / "train.jsonl"

    assert M.write_manifest(path, rows) == 1
    assert path.read_text(encoding="utf-8").count("\n") == 1  # one JSON object per line

    back = M.read_manifest(path)
    assert back[0]["text"] == "Niaje msee, leo tuna chill."
    assert back[0]["audio"] == wav.name
    assert back[0]["language"] == "sw"
    assert "__line__" in back[0]


def test_valid_manifest_is_ok(tmp_path, wav_factory):
    wav = wav_factory(tmp_path / "a.wav", seconds=5.0)
    path = tmp_path / "train.jsonl"
    M.write_manifest(
        path, [{"audio": "a.wav", "text": "Sasa hivi tunaanza.", "duration": 5.0, "language": "sw"}]
    )

    rep = M.validate_manifest(path, sample_rate=16_000)
    assert rep.ok, rep.errors
    assert rep.rows == 1
    assert round(rep.summary()["duration_hours"], 4) == round(5.0 / 3600, 4)
    assert rep.languages == {"sw": 1}
    assert wav.exists()


def test_missing_audio_and_empty_text_are_errors(tmp_path):
    path = tmp_path / "train.jsonl"
    M.write_manifest(
        path,
        [
            {"audio": "nope.wav", "text": "halo"},
            {"audio": "also_nope.wav", "text": "   "},
        ],
    )
    rep = M.validate_manifest(path)
    assert not rep.ok
    joined = " ".join(rep.errors)
    assert "audio not found" in joined
    assert "missing/empty 'text'" in joined


def test_wrong_sample_rate_is_an_error(tmp_path, wav_factory):
    """The voice encoder and the S3 tokenizer both assume 16 kHz input."""
    wav_factory(tmp_path / "a.wav", seconds=4.0, sample_rate=22_050)
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [{"audio": "a.wav", "text": "test"}])

    rep = M.validate_manifest(path, sample_rate=16_000)
    assert not rep.ok
    assert "sample rate 22050 != 16000" in rep.errors[0]
    assert M.validate_manifest(path, check_sample_rate=False).ok


def test_duration_warnings(tmp_path, wav_factory):
    wav_factory(tmp_path / "short.wav", seconds=0.5)
    wav_factory(tmp_path / "long.wav", seconds=31.0)
    path = tmp_path / "train.jsonl"
    M.write_manifest(
        path,
        [
            {"audio": "short.wav", "text": "a"},
            {"audio": "long.wav", "text": "b", "duration": 30.0},
        ],
    )
    rep = M.validate_manifest(path)
    assert rep.ok
    assert any("0.50s <" in w for w in rep.warnings)
    assert any("31.00s >" in w for w in rep.warnings)
    assert any("declared duration" in w for w in rep.warnings)


def test_language_must_be_a_string(tmp_path, wav_factory):
    wav_factory(tmp_path / "a.wav")
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [{"audio": "a.wav", "text": "x", "language": 12}])
    rep = M.validate_manifest(path)
    assert not rep.ok
    assert "'language' must be a string" in " ".join(rep.errors)


def test_report_markdown(tmp_path, wav_factory):
    wav_factory(tmp_path / "a.wav")
    path = tmp_path / "train.jsonl"
    M.write_manifest(path, [{"audio": "a.wav", "text": "x", "speaker": "abel"}])
    rep = M.validate_manifest(path)
    out = M.write_data_report(tmp_path / "report.md", [rep], extra={"dataset": "d"})
    text = out.read_text(encoding="utf-8")
    assert "# Dataset report" in text
    assert "dataset: d" in text
    assert "speakers: {'abel': 1}" in text
    assert "languages: {'sw': 1}" in text


def test_invalid_json_line_raises(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"audio": "a.wav", "text": "ok"}\nnot json\n', encoding="utf-8")
    try:
        M.read_manifest(path)
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
