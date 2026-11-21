from __future__ import annotations

import numpy as np

from sauti_sheng.resample import resample, resample_numpy, seconds, to_mono


def _tone(sr: int, hz: float, secs: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * secs), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_identity_is_free():
    x = _tone(22050, 440, 0.1)
    assert np.array_equal(resample(x, 22050, 22050), x)


def test_length_follows_ratio():
    x = _tone(22050, 440, 1.0)
    y = resample(x, 22050, 24000)
    assert abs(len(y) - 24000) <= 2


def test_numpy_fallback_matches_length():
    x = _tone(22050, 440, 1.0)
    y = resample_numpy(x, 22050, 24000)
    assert abs(len(y) - 24000) <= 2


def test_amplitude_is_preserved():
    x = _tone(22050, 300, 0.5, amp=0.4)
    y = resample(x, 22050, 24000)
    # Ignore the filter edges.
    assert 0.3 < float(np.max(np.abs(y[500:-500]))) < 0.5


def test_dc_offset_survives():
    x = np.full(22050, 0.25, dtype=np.float32)
    y = resample(x, 22050, 24000)
    assert abs(float(np.mean(y[500:-500])) - 0.25) < 0.02


def test_to_mono_averages_channels():
    x = np.stack([np.ones(10, np.float32), np.zeros(10, np.float32)], axis=1)
    assert np.allclose(to_mono(x), 0.5)


def test_seconds():
    assert seconds(24000, 24000) == 1.0
