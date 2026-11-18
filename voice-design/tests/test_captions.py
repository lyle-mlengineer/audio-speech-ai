"""Captions are the training signal - they must be truthful, varied and parseable."""

from __future__ import annotations

import re

import pytest

from kenya_voices import captions as C


def test_caption_has_no_unfilled_slots():
    attrs = C.VoiceAttrs(gender="female", age="25-35", f0_hz=190.0, words_per_sec=3.1, location="Nairobi")
    for variant in range(20):
        caption = C.build_caption(attrs, key="spk-a", variant=variant)
        assert "{" not in caption and "}" not in caption
        assert caption[0].isupper()
        assert "Kenyan" in caption or "Kenya" in caption
        assert "female" in caption.lower()


def test_caption_is_deterministic_and_varied():
    attrs = C.VoiceAttrs(gender="male", age="36-49", f0_hz=110.0)
    first = C.build_caption(attrs, key="x")
    assert first == C.build_caption(attrs, key="x")
    variants = C.caption_variants(attrs, "x", 4)
    assert len(variants) == len(set(variants)) >= 2


def test_caption_uses_only_measured_attributes():
    # No f0 / pace / age -> none of those clauses may appear.
    caption = C.build_caption(C.VoiceAttrs(gender="male"))
    assert "pitched" not in caption
    assert "pace" not in caption
    assert "thirties" not in caption


def test_unknown_gender_falls_back_to_f0():
    assert C.VoiceAttrs(f0_hz=200.0).resolved_gender == "female"
    assert C.VoiceAttrs(f0_hz=100.0).resolved_gender == "male"
    assert C.VoiceAttrs().resolved_gender == ""
    assert "speaker" in C.build_caption(C.VoiceAttrs())


@pytest.mark.parametrize(
    ("f0", "label"),
    [(90.0, "low-pitched"), (130.0, "mid-pitched"), (200.0, "bright"), (260.0, "high-pitched"), (None, "")],
)
def test_pitch_bands(f0, label):
    assert label in C.pitch_descriptor(f0)


@pytest.mark.parametrize(("wps", "label"), [(1.8, "unhurried"), (2.8, "conversational"), (3.9, "brisk"), (5.0, "fast")])
def test_pace_bands(wps, label):
    assert label in C.pace_descriptor(wps)


def test_code_switching_is_described():
    assert "Sheng" in C.language_descriptor("Swahili", english_share=0.4, swahili_share=0.2)
    assert "some English" in C.language_descriptor("Swahili", english_share=0.1)
    assert C.language_descriptor("Swahili") == "Swahili"


def test_english_share():
    assert C.english_share("Honestly speaking the budget is late") > 0.5
    assert C.english_share("Habari yako leo") == 0.0
    assert C.english_share("") == 0.0


def test_location_is_cleaned():
    attrs = C.VoiceAttrs(gender="male", location="  nairobi ")
    variants = C.caption_variants(attrs, "k", len(C.TEMPLATES))
    assert any("Nairobi" in c for c in variants)
    for bad in ("12345", "x" * 40, "Nairobi, Kenya;", ""):
        assert not any(
            "12345" in c or "Xxxx" in c for c in C.caption_variants(C.VoiceAttrs(gender="male", location=bad), "k", 8)
        )
        assert any("Kenya" in c for c in C.caption_variants(C.VoiceAttrs(gender="male", location=bad), "k", 8))


@pytest.mark.parametrize(
    "oriented",
    [
        "(A Kenyan male voice)Habari yako",
        "(A Kenyan male voice) Habari yako",
        "  (A Kenyan male voice)Habari yako",
    ],
)
def test_caption_roundtrip(oriented):
    caption, text = C.split_caption(C.caption_with_text("A Kenyan male voice", "Habari yako"))
    assert caption == C.split_caption(oriented)[0] == "A Kenyan male voice"
    assert text == C.split_caption(oriented)[1] == "Habari yako"


def test_split_caption_without_caption():
    assert C.split_caption("Habari yako") == ("", "Habari yako")
    assert C.split_caption("") == ("", "")


def test_caption_with_text_strips_trailing_period():
    assert C.caption_with_text("A Kenyan voice.", "Habari") == "(A Kenyan voice)Habari"
    assert C.caption_with_text("", "Habari") == "Habari"


def test_profile_aggregates_speaker_attributes():
    profile = C.build_profile(
        "spk1",
        gender="Female",
        age="25-35",
        location="Nairobi",
        f0_values=[180.0, 190.0, 195.0, 200.0],
        texts=["Habari yako leo", "the budget meeting is late"],
        durations=[2.0, 3.0],
    )
    assert profile.gender == "female"
    assert profile.f0_hz == pytest.approx(192.5)
    assert profile.words_per_sec == pytest.approx(8 / 5.0, abs=0.01)
    assert profile.english_share > 0
    assert profile.caption
    assert profile.clips == 2


def test_profile_refuses_to_guess_pitch_from_two_clips():
    profile = C.build_profile("spk", gender="male", f0_values=[100.0, 110.0], texts=["a b c"], durations=[1.0])
    assert profile.f0_hz is None
    assert "pitched" not in profile.caption


def test_profile_caption_is_stable_across_calls():
    kwargs = dict(f0_values=[100.0] * 5, texts=["Habari yako leo"], durations=[1.0], gender="male")
    assert C.build_profile("s", **kwargs).caption == C.build_profile("s", **kwargs).caption


def test_caption_dropout_is_deterministic():
    keys = [f"clip-{i}" for i in range(500)]
    assert not any(C.drop_caption(k, 0.0) for k in keys)
    assert all(C.drop_caption(k, 1.0) for k in keys)
    first = [C.drop_caption(k, 0.3) for k in keys]
    assert first == [C.drop_caption(k, 0.3) for k in keys]
    assert 0.15 < sum(first) / len(keys) < 0.45


def test_captions_stay_short():
    attrs = C.VoiceAttrs(gender="female", age="50+", f0_hz=210.0, words_per_sec=4.5, english_share=0.4, location="Mombasa")
    assert len(C.build_caption(attrs, key="k")) < 160
    assert not re.search(r",\s*,", C.build_caption(attrs, key="k"))
