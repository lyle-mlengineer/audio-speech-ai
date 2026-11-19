"""Manifests: the seam between the data stage and every consumer."""

from __future__ import annotations

import pytest

from sheng_voiceclone import manifest as mf


def test_write_read_roundtrip(tmp_path):
    rows = [{"audio": "a.wav", "text": "sasa bro", "speaker": "abel"}]
    p = mf.write_jsonl(tmp_path / "m.jsonl", rows)
    assert mf.read_jsonl(p) == rows


def test_read_skips_blank_lines_and_reports_bad_json(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"audio": "a.wav", "text": "x"}\n\n', encoding="utf-8")
    assert len(mf.read_jsonl(p)) == 1
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(mf.ManifestError, match="not JSON"):
        mf.read_jsonl(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(mf.ManifestError, match="not found"):
        mf.read_jsonl(tmp_path / "nope.jsonl")


def test_validate_flags_missing_keys_empty_text_and_missing_audio(tmp_path):
    (tmp_path / "ok.wav").write_bytes(b"RIFF")
    rows = [
        {"audio": "ok.wav", "text": "fine"},
        {"audio": "ok.wav"},
        {"audio": "ok.wav", "text": "   "},
        {"audio": "gone.wav", "text": "x"},
        {"text": "no audio"},
    ]
    problems = mf.validate(rows, root=tmp_path)
    assert any("missing 'text'" in p for p in problems)
    assert any("empty text" in p for p in problems)
    assert any("audio not found" in p for p in problems)
    assert any("missing 'audio'" in p for p in problems)


def test_validate_can_skip_file_checks():
    rows = [{"audio": "/nowhere/a.wav", "text": "x"}]
    assert mf.validate(rows, check_files=False) == []
    assert mf.validate(rows) != []


def test_validate_empty_manifest():
    assert mf.validate([]) == ["manifest has no rows"]


def test_load_raises_with_a_bounded_message(tmp_path):
    rows = [{"audio": "gone.wav", "text": "x"} for _ in range(30)]
    p = mf.write_jsonl(tmp_path / "m.jsonl", rows)
    with pytest.raises(mf.ManifestError) as ei:
        mf.load(p, root=tmp_path)
    msg = str(ei.value)
    assert "30 problem(s)" in msg and "more" in msg


def test_rel_audio_is_posix_and_relative(tmp_path):
    root = tmp_path / "art"
    p = root / "data" / "wav16k" / "u.wav"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"RIFF")
    assert mf.rel_audio(p, root) == "data/wav16k/u.wav"


def test_rel_audio_falls_back_to_absolute_outside_root(tmp_path):
    outside = tmp_path / "elsewhere.wav"
    outside.write_bytes(b"RIFF")
    assert mf.rel_audio(outside, tmp_path / "art") == str(outside)


def test_resolve_audio_joins_relative_paths(tmp_path):
    row = {"audio": "data/x.wav", "text": "t"}
    assert mf.resolve_audio(row, tmp_path) == tmp_path / "data" / "x.wav"
    absolute = tmp_path / "abs.wav"
    assert mf.resolve_audio({"audio": str(absolute)}, tmp_path) == absolute
