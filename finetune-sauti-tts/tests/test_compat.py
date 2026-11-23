from __future__ import annotations

import numpy as np
import pytest

from sauti_sheng.compat import AudioInfo, patch_torchaudio, seed_everything

torch = pytest.importorskip("torch")


def test_patch_is_idempotent_and_loads_a_wav(tmp_path):
    import soundfile as sf
    import torchaudio

    path = tmp_path / "a.wav"
    sf.write(str(path), np.zeros(2400, dtype=np.float32), 24000)
    assert patch_torchaudio() in (True, False)
    assert patch_torchaudio() is False  # already patched

    audio, sr = torchaudio.load(str(path))
    assert sr == 24000
    assert tuple(audio.shape) == (1, 2400)
    assert audio.dtype == torch.float32

    info = torchaudio.info(str(path))
    assert isinstance(info, AudioInfo)
    assert info.sample_rate == 24000
    assert info.num_frames == 2400
    assert info.num_channels == 1


def test_patch_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SAUTI_KEEP_TORCHAUDIO", "1")
    assert patch_torchaudio(force=True) is False


def test_seed_everything_is_deterministic():
    seed_everything(7)
    a = torch.rand(3)
    seed_everything(7)
    b = torch.rand(3)
    assert torch.equal(a, b)
