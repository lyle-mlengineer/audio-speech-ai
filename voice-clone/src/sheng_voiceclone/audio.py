"""Audio IO and sample-rate conversion with no heavy dependency.

Every clone-model in this repo consumes 16 kHz mono (Chatterbox: voice encoder +
S3 tokenizer) while the source dataset is 22.05 kHz. So audio is converted once,
at data-prep time, and never again.

``scipy.signal.resample_poly`` is used only if it is already installed; the numpy
polyphase fallback is the same algorithm and keeps the data stage runnable (and
testable) on a bare Python install. The filter bank is normalised to unity DC
gain, so a resampled 1 kHz tone keeps its amplitude.
"""

from __future__ import annotations

import math
from math import gcd
from pathlib import Path

import numpy as np

# Taps per polyphase branch, so the filter is 2 * HALF_TAPS * up long.
HALF_TAPS = 16
KAISER_BETA = 8.0


def to_mono(x: np.ndarray) -> np.ndarray:
    """(n,) or (n, ch) float array -> (n,) float32."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x.reshape(-1), dtype=np.float32)


def _polyphase_matrix(up: int, down: int, cutoff: float) -> np.ndarray:
    """(up, 2*HALF_TAPS+1) filter bank, each row normalised to unity DC gain."""
    half = HALF_TAPS
    offset = half * up
    j = np.arange(-offset, offset + up, dtype=np.float64)
    h = cutoff * np.sinc(cutoff * j / up) * np.kaiser(j.size, KAISER_BETA)
    phases = np.stack([h[offset + p + up * np.arange(-half, half + 1)] for p in range(up)])
    gains = phases.sum(axis=1, keepdims=True)
    gains[gains == 0] = 1.0
    return phases / gains


def _resample_numpy(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
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
    idx0 = (prod - p) // up

    half = HALF_TAPS
    pad = half + 1
    padded = np.concatenate([np.zeros(pad), x, np.zeros(pad + up)])
    base = idx0 + pad

    y = np.zeros(n_out, dtype=np.float64)
    taps = np.arange(-half, half + 1)
    for phase in range(up):
        sel = np.nonzero(p == phase)[0]
        if sel.size == 0:
            continue
        row = phases[phase]
        idx = base[sel][:, None] - taps[None, :]
        y[sel] = (padded[idx] * row[None, :]).sum(axis=1)
    return y.astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited resampling by the rational factor ``sr_out / sr_in``."""
    if int(sr_in) == int(sr_out):
        return to_mono(x)
    try:  # scipy is ~50x faster and is always present on Colab
        from scipy.signal import resample_poly

        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(to_mono(x), int(sr_out) // g, int(sr_in) // g).astype(np.float32)
    except Exception:
        return _resample_numpy(x, sr_in, sr_out)


# ------------------------------------------------------------------------ IO
def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode any soundfile-readable file to mono float32 in [-1, 1]."""
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono(x), int(sr)


def read_16k(path: str | Path) -> np.ndarray:
    """Decode and convert to 16 kHz mono float32 (what every model here wants)."""
    x, sr = read_wav(path)
    return resample(x, sr, 16000)


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> Path:
    """Write mono PCM16, clipped, creating parents."""
    import soundfile as sf

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), np.clip(to_mono(x), -1.0, 1.0), int(sr), subtype="PCM_16")
    return p


def duration(path: str | Path) -> float:
    """Duration in seconds, from the container header (no full decode)."""
    import soundfile as sf

    return float(sf.info(str(path)).duration)


def stats(x: np.ndarray, sr: int) -> dict[str, float]:
    """RMS / peak / clipping / silence facts used by the scorer."""
    x = to_mono(x)
    if x.size == 0:
        return {"seconds": 0.0, "rms": 0.0, "peak": 0.0, "clipped_frac": 0.0, "silent_frac": 1.0}
    a = np.abs(x)
    return {
        "seconds": float(x.size) / float(sr),
        "rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
        "peak": float(a.max()),
        # PCM16 clips at 1.0; anything at/over it for more than 1% of samples is bad
        "clipped_frac": float(np.mean(a >= 0.999)),
        "silent_frac": float(np.mean(a < 1e-4)),
    }
