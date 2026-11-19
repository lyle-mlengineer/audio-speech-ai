import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture()
def art(tmp_path):
    from sheng_voiceclone.paths import Paths

    p = Paths(root=tmp_path / "art")
    p.ensure_dirs()
    return p


@pytest.fixture()
def tone():
    """A 1 kHz-ish tone generator at any sample rate, 16-bit safe."""
    import numpy as np

    def make(seconds=1.0, sr=16000, freq=220.0, amp=0.3):
        t = np.arange(int(seconds * sr), dtype=np.float32) / float(sr)
        return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)

    return make


@pytest.fixture()
def wav(tmp_path):
    """Write a tone to disk and return the path."""
    import soundfile as sf

    def make(name="a.wav", seconds=1.0, sr=16000, amp=0.3, freq=220.0):
        import numpy as np

        t = np.arange(int(seconds * sr), dtype=np.float32) / float(sr)
        x = (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), x, sr, subtype="PCM_16")
        return p

    return make


@pytest.fixture()
def rows_file(art):
    """A valid 3-row manifest inside an artifacts root."""
    from sheng_voiceclone import manifest as mf

    def make(n=3, text="sasa bro uko aje"):
        import soundfile as sf
        import numpy as np

        rows = []
        for i in range(1, n + 1):
            p = art.wav_dir / f"utt_{i:04d}.wav"
            sf.write(str(p), np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
            rows.append({"audio": mf.rel_audio(p, art.root), "text": f"{text} {i}", "speaker": "abel"})
        mf.write_jsonl(art.train_manifest, rows)
        return rows

    return make


@pytest.fixture()
def scores_file(art):
    """Write a scores/<name>.json the way the CLI does."""
    def make(name, cer, sim, ok=5, clips=7):
        payload = {
            "name": name,
            "summary": {
                "clips": clips,
                "ok": ok,
                "statuses": {"ok": ok},
                "cer_mean": cer,
                "wer_mean": None if cer is None else cer * 2,
                "similarity_mean": sim,
                "seconds_total": 30.0,
            },
            "rows": [],
        }
        p = art.scores / f"{name}.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    return make
