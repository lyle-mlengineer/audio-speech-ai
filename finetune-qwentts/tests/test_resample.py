from __future__ import annotations

import numpy as np
import pytest

from sheng_qwentts.resample import resample_numpy, to_mono


def _tone(freq: float, sr: int, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.mark.parametrize("sr_in,sr_out", [(22050, 24000), (16000, 24000), (44100, 24000), (24000, 24000)])
def test_length(sr_in: int, sr_out: int):
    x = _tone(440, sr_in)
    y = resample_numpy(x, sr_in, sr_out)
    assert y.size == int(np.floor(x.size * sr_out / sr_in))


@pytest.mark.parametrize("freq", [100.0, 1000.0, 3000.0])
def test_preserves_level_and_frequency(freq: float):
    sr_in, sr_out = 22050, 24000
    x = _tone(freq, sr_in)
    y = resample_numpy(x, sr_in, sr_out)
    inner = y[2000:-2000]
    assert inner.std() == pytest.approx(x.std(), rel=0.02)

    spec = np.abs(np.fft.rfft(inner * np.hanning(inner.size)))
    freqs = np.fft.rfftfreq(inner.size, 1 / sr_out)
    assert freqs[spec.argmax()] == pytest.approx(freq, abs=2.0)


def test_identity_is_a_noop():
    x = _tone(440, 24000, 0.1)
    assert np.array_equal(resample_numpy(x, 24000, 24000), x)


def test_no_nan_and_finite():
    rng = np.random.default_rng(0)
    y = resample_numpy(rng.normal(0, 0.1, 2205).astype(np.float32), 22050, 24000)
    assert np.isfinite(y).all()
    assert y.dtype == np.float32


def test_to_mono():
    stereo = np.stack([np.ones(10, dtype=np.float32), -np.ones(10, dtype=np.float32)], axis=1)
    assert np.allclose(to_mono(stereo), 0.0)
    mono = np.ones(10, dtype=np.float32)
    assert to_mono(mono).shape == (10,)
