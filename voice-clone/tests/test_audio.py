"""Resampling and IO: the 22.05 -> 16 kHz step must not change the signal."""

from __future__ import annotations

import numpy as np
import pytest

from sheng_voiceclone import audio


def tone(seconds, sr, freq=440.0, amp=0.5):
    t = np.arange(int(seconds * sr), dtype=np.float64) / float(sr)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_to_mono_downmixes_and_never_returns_2d():
    x = np.zeros((100, 2), dtype=np.float32)
    x[:, 0] = 0.5
    x[:, 1] = -0.5
    out = audio.to_mono(x)
    assert out.ndim == 1 and out.dtype == np.float32
    assert np.allclose(out, 0.0)


@pytest.mark.parametrize("sr_in,sr_out", [(22050, 16000), (16000, 16000), (48000, 16000)])
def test_resample_length_and_unity_gain(sr_in, sr_out):
    x = tone(0.5, sr_in, freq=1000.0, amp=0.5)
    y = audio.resample(x, sr_in, sr_out)
    assert abs(y.size - int(0.5 * sr_out)) <= 2
    # 1 kHz is well inside the passband: amplitude must survive the filter
    steady = y[int(0.1 * sr_out) : int(0.4 * sr_out)]
    assert np.max(np.abs(steady)) == pytest.approx(0.5, abs=0.06)


def test_resample_passthrough_is_identity_shape():
    x = tone(0.1, 16000)
    assert audio.resample(x, 16000, 16000).shape == x.shape


def test_resample_empty_is_empty():
    assert audio.resample(np.zeros(0, dtype=np.float32), 22050, 16000).size == 0


def test_write_and_read_roundtrip(tmp_path):
    x = tone(0.25, 16000, amp=0.4)
    p = audio.write_wav(tmp_path / "sub" / "x.wav", x, 16000)
    assert p.is_file()
    back, sr = audio.read_wav(p)
    assert sr == 16000 and back.size == x.size
    assert audio.duration(p) == pytest.approx(0.25, abs=0.01)


def test_read_16k_converts_on_the_fly(wav):
    p = wav("src22k.wav", seconds=0.2, sr=22050)
    x = audio.read_16k(p)
    assert abs(x.size - int(0.2 * 16000)) <= 2


def test_write_clips_instead_of_wrapping():
    x = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    import soundfile as sf

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as d:
        p = audio.write_wav(f"{d}/clip.wav", x, 16000)
        back, _ = sf.read(str(p), dtype="float32")
    assert back.max() <= 1.0 and back.min() >= -1.0


def test_stats_flags_silence_and_clipping():
    quiet = audio.stats(np.zeros(1600, dtype=np.float32), 16000)
    assert quiet["rms"] == 0.0 and quiet["silent_frac"] == 1.0
    loud = audio.stats(np.ones(1600, dtype=np.float32), 16000)
    assert loud["clipped_frac"] == 1.0 and loud["rms"] == pytest.approx(1.0)
    empty = audio.stats(np.zeros(0, dtype=np.float32), 16000)
    assert empty["seconds"] == 0.0
