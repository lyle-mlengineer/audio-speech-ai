"""Sample-rate conversion with no heavy dependency.

The Qwen3-TTS tokenizer, the speaker encoder and the training mel are all
24 kHz-only (``finetuning/dataset.py`` in the upstream repo asserts
``sr == 24000``), while the parquet holds 22.05 kHz audio. So every clip has to
be converted once, at data-prep time.

``scipy.signal.resample_poly`` is used when scipy is around (it always is on
Colab, librosa pulls it in). The numpy polyphase fallback exists so the data
stage runs and is testable on a bare Python install - it is the same algorithm,
just slower.
"""

from __future__ import annotations

import math
from math import gcd

import numpy as np

# Taps per polyphase branch (so the filter is 2 * HALF_TAPS * up long in total).
HALF_TAPS = 16
KAISER_BETA = 8.0


def _polyphase_matrix(up: int, down: int, cutoff: float) -> np.ndarray:
    """(up, 2*HALF_TAPS+1) filter bank, each row normalised to unity DC gain."""
    half = HALF_TAPS
    offset = half * up
    # One extra branch worth of taps at the top so every phase has 2*half+1 of them.
    j = np.arange(-offset, offset + up, dtype=np.float64)
    # Ideal lowpass in input-sample units, Kaiser-windowed.
    h = cutoff * np.sinc(cutoff * j / up) * np.kaiser(j.size, KAISER_BETA)
    # h[j] with j = p + k*up  ->  row p, column k + half
    phases = np.stack([h[offset + p + up * np.arange(-half, half + 1)] for p in range(up)])
    gains = phases.sum(axis=1, keepdims=True)
    gains[gains == 0] = 1.0
    return phases / gains


def resample_numpy(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited resampling by the rational factor sr_out/sr_in."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    if up == 1 and down == 1:
        return x.astype(np.float32)

    cutoff = min(1.0, sr_out / sr_in)
    phases = _polyphase_matrix(up, down, cutoff)
    n_out = int(math.floor(x.size * up / down))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    m = np.arange(n_out, dtype=np.int64)
    prod = m * down
    p = prod % up
    idx0 = (prod - p) // up  # input sample under the filter centre

    half = HALF_TAPS
    # Zero-pad the input so the tap gathering never needs bounds checks.
    pad = half + 1
    padded = np.concatenate([np.zeros(pad), x, np.zeros(pad + up)])
    base = idx0 + pad

    y = np.zeros(n_out, dtype=np.float64)
    taps = np.arange(-half, half + 1)
    for phase in range(up):
        sel = np.nonzero(p == phase)[0]
        if sel.size == 0:
            continue
        row = phases[phase]  # (2*half+1,)
        idx = base[sel][:, None] - taps[None, :]
        y[sel] = (padded[idx] * row[None, :]).sum(axis=1)
    return y.astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono float audio, scipy when available else the numpy fallback."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if int(sr_in) == int(sr_out):
        return x
    try:
        from scipy.signal import resample_poly  # type: ignore

        g = gcd(int(sr_in), int(sr_out))
        out = resample_poly(x, int(sr_out) // g, int(sr_in) // g)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        return resample_numpy(x, sr_in, sr_out)


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def seconds(n_samples: int, sr: int) -> float:
    return float(n_samples) / float(sr)
