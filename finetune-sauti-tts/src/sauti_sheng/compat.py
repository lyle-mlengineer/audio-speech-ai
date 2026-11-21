"""Colab-proof audio I/O and other small runtime shims.

The one thing that reliably breaks F5-TTS on a fresh Colab VM is
``torchaudio.load``: recent torchaudio releases delegate decoding to
``torchcodec``, which wants FFmpeg shared libraries that Colab does not ship in
the right place. F5's training loader and its inference path both call it, so
the failure surfaces as a ``torchcodec``/``libav*`` error in the middle of a
run.

``soundfile`` (libsndfile) is already a dependency of every stage and reads the
16-bit PCM WAVs this pipeline writes without complaint. So we swap the three
functions F5 actually uses - ``load``, ``save`` and ``info`` - for soundfile
implementations. ``torchaudio.transforms.Resample`` is untouched: it is pure
torch.

The patch is idempotent and can be disabled with ``SAUTI_KEEP_TORCHAUDIO=1``
if you would rather debug the real torchaudio stack.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

PATCH_FLAG = "__sauti_sheng_patched__"
KEEP_ENV = "SAUTI_KEEP_TORCHAUDIO"


@dataclass(frozen=True)
class AudioInfo:
    """The subset of ``torchaudio.info`` that F5 and friends read."""

    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"

    @property
    def num_samples(self) -> int:
        return self.num_frames


def _sf_load(
    filepath,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    **_kwargs,
):
    """Drop-in ``torchaudio.load`` returning float32 ``(channels, frames)``."""
    import numpy as np
    import soundfile as sf
    import torch

    start = max(0, int(frame_offset or 0))
    frames = -1 if num_frames in (None, -1) else max(0, int(num_frames))
    data, sample_rate = sf.read(
        str(filepath), start=start, frames=frames, dtype="float32", always_2d=True
    )
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    if not channels_first:
        waveform = waveform.T
    # ``normalize=False`` would ask for integer samples; float32 is what every
    # caller in F5 expects, so it is deliberately ignored.
    return waveform, int(sample_rate)


def _sf_save(filepath, src, sample_rate: int, **_kwargs):
    """Drop-in ``torchaudio.save`` accepting a tensor or an array."""
    import numpy as np
    import soundfile as sf

    try:
        import torch

        if isinstance(src, torch.Tensor):
            array = src.detach().cpu().float().numpy()
        else:
            array = np.asarray(src, dtype=np.float32)
    except Exception:  # pragma: no cover - torch is present whenever this runs
        array = np.asarray(src, dtype=np.float32)

    if array.ndim == 2:
        array = array.T  # (channels, frames) -> (frames, channels)
    sf.write(str(filepath), array, int(sample_rate))
    return None


def _sf_info(filepath, **_kwargs) -> AudioInfo:
    import soundfile as sf

    info = sf.info(str(filepath))
    return AudioInfo(
        sample_rate=int(info.samplerate),
        num_frames=int(info.frames),
        num_channels=int(info.channels),
        bits_per_sample=16,
        encoding=str(info.subtype or "PCM_S"),
    )


def patch_torchaudio(force: bool = False) -> bool:
    """Replace ``torchaudio.load/save/info`` with soundfile versions.

    Returns True when a patch was applied by this call. Safe to call from every
    entry point (training, inference, export): the flag makes it a no-op the
    second time.
    """
    if os.environ.get(KEEP_ENV):
        return False
    try:
        import torchaudio
    except Exception:
        return False
    if getattr(torchaudio, PATCH_FLAG, False) and not force:
        return False
    torchaudio.load = _sf_load
    torchaudio.save = _sf_save
    torchaudio.info = _sf_info
    setattr(torchaudio, PATCH_FLAG, True)
    return True


def seed_everything(seed: int = 1337) -> None:
    """Seed python, numpy and torch without importing torch when absent."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def free_memory() -> None:
    """Release CPU/GPU memory after a 5 GB checkpoint has been inspected."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
