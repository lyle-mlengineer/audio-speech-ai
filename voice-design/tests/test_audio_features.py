"""Audio measurements keep captions honest: pitch and pace must come from the audio."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kenya_voices.audio_features import MAX_F0, MIN_F0, median_f0, words_per_second


def sine(f0: float, seconds: float = 2.0, sample_rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.4 * np.sin(2 * math.pi * f0 * t)).astype("float32")


@pytest.mark.parametrize("f0", [85.0, 120.0, 190.0, 240.0])
def test_median_f0_tracks_a_sine(f0):
    assert median_f0(sine(f0), 16_000) == pytest.approx(f0, abs=6.0)


def test_median_f0_survives_noise():
    rng = np.random.default_rng(0)
    signal = sine(110.0) + rng.normal(0, 0.02, int(2.0 * 16_000)).astype("float32")
    assert median_f0(signal, 16_000) == pytest.approx(110.0, abs=8.0)


def test_median_f0_of_silence_is_none():
    assert median_f0(np.zeros(16_000, dtype="float32"), 16_000) is None
    assert median_f0(np.zeros(10, dtype="float32"), 16_000) is None


def test_median_f0_ignores_out_of_band_content():
    # 20 Hz is below MIN_F0 and 800 Hz above MAX_F0: nothing voiced in band.
    assert median_f0(sine(20.0), 16_000) is None
    assert median_f0(sine(800.0), 16_000) is None
    assert MIN_F0 < 100 < MAX_F0


def test_median_f0_respects_sample_rate():
    assert median_f0(sine(150.0, sample_rate=48_000), 48_000) == pytest.approx(150.0, abs=8.0)


def test_words_per_second():
    assert words_per_second("habari yako leo rafiki", 2.0) == pytest.approx(2.0)
    assert words_per_second("habari yako", 2.0) is None  # too few words to be a rate
    assert words_per_second("", 2.0) is None
    assert words_per_second("a b c", 0) is None


def test_accepts_lists():
    assert median_f0(list(sine(130.0)), 16_000) == pytest.approx(130.0, abs=6.0)
