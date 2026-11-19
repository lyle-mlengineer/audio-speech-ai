"""Eval plumbing that must work with no GPU: plans, offline scoring, compare."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from sheng_voiceclone import evals, metrics, prompts


def _write(path, seconds=2.0, sr=16000, amp=0.3, freq=200.0):
    t = np.arange(int(seconds * sr), dtype=np.float32) / float(sr)
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="PCM_16")
    return path


# ------------------------------------------------------------------- prompts
def test_prompt_buckets_cover_sheng_swahili_english():
    buckets = {p["bucket"] for p in prompts.PROMPTS}
    assert buckets == {"sheng", "swahili", "english"}
    assert len({p["id"] for p in prompts.PROMPTS}) == len(prompts.PROMPTS)


def test_prompt_selection_by_bucket_id_and_unknown():
    assert {p["bucket"] for p in prompts.select("swahili")} == {"swahili"}
    assert [p["id"] for p in prompts.select("sheng_01")] == ["sheng_01"]
    assert len(prompts.select("sheng_01,sheng_01")) == 1  # deduplicated
    with pytest.raises(KeyError, match="unknown prompt id"):
        prompts.select("nope")


# ---------------------------------------------------------------------- plans
def test_plan_and_write_plan(tmp_path):
    rows = evals.plan(prompts.select("sheng_01"), tmp_path / "base", tag="baseline")
    assert rows[0]["clip"] == "sheng_01"
    assert rows[0]["wav"].endswith("sheng_01.wav")
    assert rows[0]["expected_seconds"] > 1.0
    p = evals.write_plan(rows, tmp_path / "base")
    assert json.loads(p.read_text(encoding="utf-8"))[0]["tag"] == "baseline"


# -------------------------------------------------------------------- scoring
def test_score_rows_with_perfect_judges_is_ok(art, tmp_path):
    rows = evals.plan(prompts.select("sheng_01"), tmp_path / "ev", tag="t")
    _write(tmp_path / "ev" / "sheng_01.wav", seconds=rows[0]["expected_seconds"])
    scored, summary = evals.score_rows(
        rows,
        ref_audio=_write(tmp_path / "ref.wav"),
        transcribe=lambda p: rows[0]["text"],   # perfect ASR
        embed=lambda p: [1.0, 0.0, 0.0],        # same speaker
    )
    assert scored[0]["cer"] == 0.0 and scored[0]["similarity"] == pytest.approx(1.0)
    assert scored[0]["status"] == "ok"
    assert summary["ok"] == 1


def test_score_rows_without_judges_still_checks_the_waveform(art, tmp_path):
    rows = evals.plan(prompts.select("sheng_01"), tmp_path / "ev", tag="t")
    _write(tmp_path / "ev" / "sheng_01.wav", seconds=2.0, amp=0.0)
    scored, _ = evals.score_rows(rows, ref_audio=_write(tmp_path / "ref.wav"))
    assert scored[0]["status"] == "silent"
    assert scored[0]["cer"] is None and scored[0]["similarity"] is None


def test_score_rows_marks_a_missing_wav(art, tmp_path):
    rows = evals.plan(prompts.select("sheng_01"), tmp_path / "empty", tag="t")
    scored, summary = evals.score_rows(rows, ref_audio=_write(tmp_path / "ref.wav"))
    assert scored[0]["status"] == "missing"
    assert summary["clips"] == 1 and summary["ok"] == 0


def test_score_rows_catches_wrong_words(art, tmp_path):
    rows = evals.plan(prompts.select("sheng_01"), tmp_path / "ev", tag="t")
    _write(tmp_path / "ev" / "sheng_01.wav", seconds=rows[0]["expected_seconds"])
    scored, _ = evals.score_rows(
        rows,
        ref_audio=_write(tmp_path / "ref.wav"),
        transcribe=lambda p: "kitu kingine kabisa",
        embed=lambda p: [1.0, 0.0],
    )
    assert scored[0]["status"] == "wrong-words"


def test_write_and_read_scores_roundtrip(art, tmp_path):
    rows = evals.plan(prompts.select("sheng_01"), tmp_path / "ev", tag="t")
    _write(tmp_path / "ev" / "sheng_01.wav", seconds=2.0)
    scored, summary = evals.score_rows(rows, ref_audio=_write(tmp_path / "ref.wav"))
    js, md = evals.write_scores(art, "baseline", scored, summary, title="Base")
    assert js.is_file() and md.is_file() and "Base" in md.read_text(encoding="utf-8")
    back = evals.read_scores(art, "baseline")
    assert back["summary"]["clips"] == 1 and back["rows"][0]["clip"] == "sheng_01"


def test_read_missing_scores_says_what_to_run(art):
    with pytest.raises(FileNotFoundError, match="run `baseline`"):
        evals.read_scores(art, "tuned")


# -------------------------------------------------------------------- compare
def _scores(art, name, cer, sim, ok=7, wer=None):
    rows, summary = [], {
        "clips": 7,
        "ok": ok,
        "statuses": {"ok": ok},
        "cer_mean": cer,
        "wer_mean": wer if wer is not None else (None if cer is None else cer * 2),
        "similarity_mean": sim,
        "seconds_total": 40.0,
    }
    (art.scores / f"{name}.json").write_text(
        json.dumps({"name": name, "summary": summary, "rows": rows}), encoding="utf-8"
    )
    return evals.read_scores(art, name)


def test_compare_keep_when_closer_voice_and_same_words(art):
    text = evals.compare(_scores(art, "baseline", 0.30, 0.55), _scores(art, "tuned", 0.28, 0.72))
    assert "KEEP" in text and "| metric | baseline | tuned | delta | better |" in text


def test_compare_mixed_when_words_get_worse(art):
    text = evals.compare(_scores(art, "baseline", 0.20, 0.50), _scores(art, "tuned", 0.55, 0.80))
    assert "MIXED" in text


def test_compare_reject_when_more_clips_break(art):
    text = evals.compare(_scores(art, "baseline", 0.20, 0.60, ok=7), _scores(art, "tuned", 0.30, 0.50, ok=3))
    assert "REJECT" in text


def test_compare_no_gain_when_nothing_moves(art):
    text = evals.compare(_scores(art, "baseline", 0.20, 0.60), _scores(art, "tuned", 0.21, 0.59))
    assert "NO GAIN" in text


def test_compare_without_judges_refuses_to_guess(art):
    text = evals.compare(_scores(art, "baseline", None, None), _scores(art, "tuned", None, None))
    assert "NOT ENOUGH DATA" in text


def test_fingerprint_is_stable_and_short():
    assert evals.fingerprint("a", 1) == evals.fingerprint("a", 1)
    assert evals.fingerprint("a", 1) != evals.fingerprint("a", 2)
    assert len(evals.fingerprint("a", 1)) == 8
