"""The scorer is the whole point of this project, so it gets the most tests."""

from __future__ import annotations

import pytest

from sheng_voiceclone import metrics


# ------------------------------------------------------------------ text prep
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sasa bro, uko aje?", "sasa bro uko aje"),
        ("  Msee   wangu!! ", "msee wangu"),
        ("Niaje\u2019 wewe", "niaje' wewe"),
        ("Bei ni 50 bob.", "bei ni 50 bob"),
        ("", ""),
    ],
)
def test_normalize_text(raw, expected):
    assert metrics.normalize_text(raw) == expected


def test_words_and_chars_agree_on_spaces():
    s = "Sasa bro, uko aje?"
    assert metrics.words(s) == ["sasa", "bro", "uko", "aje"]
    assert "".join(metrics.chars(s)) == "sasabroukoaje"


# ----------------------------------------------------------------- edit counts
def test_edit_distance_basics():
    assert metrics.edit_distance([], []) == 0
    assert metrics.edit_distance("abc", "abc") == 0
    assert metrics.edit_distance([], ["a", "b"]) == 2
    assert metrics.edit_distance(["a", "b"], []) == 2
    assert metrics.edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert metrics.edit_distance("kitten", "sitting") == 3


def test_wer_cer_exact_and_none():
    assert metrics.wer("sasa bro uko aje", "sasa bro uko aje") == 0.0
    assert metrics.wer("sasa bro uko aje", "sasa bro uko") == pytest.approx(0.25)
    assert metrics.wer("", "anything") is None
    assert metrics.cer("", "") is None
    assert metrics.cer("abc", "abc") == 0.0


def test_wer_ignores_case_and_punctuation():
    assert metrics.wer("Sasa bro, uko aje?", "sasa BRO uko aje") == 0.0


# --------------------------------------------------------------- speaker match
def test_cosine():
    assert metrics.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert metrics.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert metrics.cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert metrics.cosine([0.0, 0.0], [1.0, 1.0]) is None
    assert metrics.cosine([1.0], [1.0, 2.0]) is None


# ------------------------------------------------------------------- verdicts
GOOD = {"seconds": 6.0, "rms": 0.1, "peak": 0.8, "clipped_frac": 0.0, "silent_frac": 0.0}


def test_judge_ok_when_everything_is_fine():
    assert metrics.judge(GOOD, expected_seconds=5.0, cer=0.1, similarity=0.8)[0] == "ok"


def test_judge_silence_wins_over_every_other_signal():
    silent = dict(GOOD, rms=0.0001)
    status, note = metrics.judge(silent, expected_seconds=5.0, cer=0.9, similarity=0.1)
    assert status == "silent" and "nothing synthesised" in note


def test_judge_clipping():
    assert metrics.judge(dict(GOOD, clipped_frac=0.2), expected_seconds=5.0)[0] == "clipping"


def test_judge_runaway_and_truncated():
    assert metrics.judge(dict(GOOD, seconds=30.0), expected_seconds=5.0)[0] == "runaway"
    assert metrics.judge(dict(GOOD, seconds=1.0), expected_seconds=5.0)[0] == "truncated"


def test_judge_words_then_voice_order():
    # bad words beat a good voice
    assert metrics.judge(GOOD, expected_seconds=5.0, cer=0.9, similarity=0.9)[0] == "wrong-words"
    # good words but a different speaker
    assert metrics.judge(GOOD, expected_seconds=5.0, cer=0.1, similarity=0.2)[0] == "wrong-voice"


def test_judge_without_reference_numbers():
    assert metrics.judge(GOOD)[0] == "ok"


def test_expected_seconds_scales_with_text():
    short = metrics.expected_seconds("Sasa bro.")
    long = metrics.expected_seconds("Sasa bro. " * 20)
    assert short == pytest.approx(1.0)  # floor
    assert long > 5.0


# ------------------------------------------------------------------ summarise
def _row(clip, cer=None, sim=None, seconds=5.0, status="ok"):
    return {"clip": clip, "cer": cer, "similarity": sim, "seconds": seconds, "status": status, "note": ""}


def test_summarise_means_and_counts():
    rows = [
        _row("a", cer=0.1, sim=0.8),
        _row("b", cer=0.3, sim=0.6),
        _row("c", status="silent", seconds=0.0),
    ]
    s = metrics.summarise(rows)
    assert s["clips"] == 3 and s["ok"] == 2
    assert s["cer_mean"] == pytest.approx(0.2)
    assert s["similarity_mean"] == pytest.approx(0.7)
    assert s["statuses"]["silent"] == 1
    assert s["seconds_total"] == pytest.approx(10.0)


def test_summarise_empty_is_all_none():
    s = metrics.summarise([])
    assert s["clips"] == 0 and s["cer_mean"] is None and s["similarity_mean"] is None


def test_render_scores_md_has_a_row_per_clip():
    rows = [_row("sheng_01", cer=0.2, sim=0.7), _row("sheng_02", status="silent", seconds=0.0)]
    md = metrics.render_scores_md("t", rows, metrics.summarise(rows))
    assert "| sheng_01 |" in md and "| sheng_02 |" in md
    assert "silent" in md and "CER mean" in md
