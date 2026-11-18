"""Audio feature measurement used to keep captions truthful.

Only what we can measure goes into a caption: median f0 (pitch today) and words
per second (pace). Both are cheap: a numpy autocorrelation pitch tracker over a
few short windows, and the transcript's word count.

Using ``librosa.pyin`` would be more accurate and ~2 orders of magnitude slower;
for caption *bands* (low/mid/bright) plain autocorrelation is enough. Rows that
fail measurement simply lose the pitch clause instead of getting a fake label.
"""

from __future__ import annotations

import numpy as np

FRAME = 1024  # 64 ms at 16 kHz
HOP = 512
MIN_F0 = 60.0
MAX_F0 = 400.0
VOICING_THRESHOLD = 0.3  # normalised autocorrelation peak


def median_f0(samples, sample_rate: int = 16_000, max_frames: int = 200) -> float | None:
    """Median f0 in Hz over voiced frames, or ``None`` when nothing is voiced."""
    arr = np.asarray(samples, dtype="float32").reshape(-1)
    if arr.size < FRAME * 2:
        return None
    arr = arr - float(arr.mean())
    peak = float(np.max(np.abs(arr)))
    if peak <= 0:
        return None
    arr = arr / peak

    starts = range(0, arr.size - FRAME, HOP)
    total = max(1, (arr.size - FRAME) // HOP + 1)
    step = max(1, total // max_frames)
    lo = int(sample_rate / MAX_F0)
    hi = int(sample_rate / MIN_F0)

    estimates: list[float] = []
    window = np.hanning(FRAME).astype("float32")
    for idx, start in enumerate(starts):
        if idx % step:
            continue
        frame = arr[start : start + FRAME] * window
        energy = float(np.dot(frame, frame))
        if energy < 1e-6:
            continue
        spectrum = np.fft.rfft(frame, 2 * FRAME)
        ac = np.fft.irfft(spectrum * np.conjugate(spectrum))[:FRAME]
        ac = ac / (ac[0] + 1e-9)
        segment = ac[lo:hi]
        if segment.size == 0:
            continue
        lag = int(np.argmax(segment)) + lo
        if ac[lag] < VOICING_THRESHOLD:
            continue
        # Parabolic refinement around the peak.
        if 0 < lag < ac.size - 1:
            a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
            denom = a - 2 * b + c
            shift = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
            lag = lag + max(-1.0, min(1.0, shift))
        f0 = sample_rate / lag
        if MIN_F0 <= f0 <= MAX_F0:
            estimates.append(float(f0))

    if len(estimates) < 5:
        return None
    return round(float(np.median(estimates)), 1)


def words_per_second(text: str, duration_sec: float) -> float | None:
    if not text or not duration_sec or duration_sec <= 0:
        return None
    words = len([w for w in text.split() if w.strip()])
    if words < 3:
        return None
    return round(words / duration_sec, 2)
