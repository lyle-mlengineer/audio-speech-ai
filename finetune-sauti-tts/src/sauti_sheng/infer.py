"""Stage 5/6 building blocks: load a Sauti checkpoint and speak.

Deliberately does **not** import ``f5_tts.infer.utils_infer``. That module pulls
in ``transformers``, ``pydub`` (needs ffmpeg), ``matplotlib`` and the ASR
pipeline at import time, none of which this pipeline needs. The few pieces that
matter - Vocos loading, text chunking, the duration heuristic and the sample
call - are reproduced here, mirroring upstream so the output stays comparable.

Sheng is Latin script, so ``convert_char_to_pinyin`` is a no-op for this data
(verified on all 42 transcripts). It is still applied, because that is what the
base model was trained and evaluated with.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .compat import patch_torchaudio
from .paths import Paths

TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256
TARGET_RMS = 0.1
CROSS_FADE_SECONDS = 0.15
NFE_STEP = 32
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0


# ------------------------------------------------------------------- vocoder
def load_vocoder(device: str = "cuda"):
    """Load the 24 kHz Vocos vocoder F5 v1 Base uses."""
    import torch
    from huggingface_hub import hf_hub_download
    from vocos import Vocos

    config_path = hf_hub_download(repo_id="charactr/vocos-mel-24khz", filename="config.yaml")
    model_path = hf_hub_download(repo_id="charactr/vocos-mel-24khz", filename="pytorch_model.bin")
    vocoder = Vocos.from_hparams(config_path)
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    from vocos.feature_extractors import EncodecFeatures

    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        state.update(
            {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
        )
    vocoder.load_state_dict(state)
    return vocoder.eval().to(device)


# ---------------------------------------------------------------------- text
_SENTENCE = re.compile(r"(?<=[;:,.!?])\s+")


def chunk_text(text: str, max_chars: int = 135) -> list[str]:
    """Split generated text so each chunk stays inside F5's ~30 s window."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(text):
        if not sentence:
            continue
        if len((current + " " + sentence).encode("utf-8")) <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks or [text]


def to_model_text(text: str) -> str:
    """Apply the same character transform F5's inference path applies."""
    from f5_tts.model.utils import convert_char_to_pinyin

    return "".join(convert_char_to_pinyin([text])[0])


# --------------------------------------------------------------------- audio
def _as_mono(audio, sample_rate: int, device):
    import torch
    import torchaudio

    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
    audio = audio.float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(int(sample_rate), TARGET_SAMPLE_RATE)(audio)
    return audio.to(device)


def crossfade(waves: list[np.ndarray], sample_rate: int = TARGET_SAMPLE_RATE,
              seconds: float = CROSS_FADE_SECONDS) -> np.ndarray:
    """Concatenate chunks with a short equal-power-ish cross-fade."""
    if not waves:
        return np.zeros(0, dtype=np.float32)
    out = waves[0]
    for nxt in waves[1:]:
        n = min(int(seconds * sample_rate), len(out), len(nxt))
        if n <= 0:
            out = np.concatenate([out, nxt])
            continue
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
        overlap = out[-n:] * fade_out + nxt[:n] * fade_in
        out = np.concatenate([out[:-n], overlap, nxt[n:]])
    return out.astype(np.float32)


def synthesise(
    model,
    vocoder,
    ref_audio,
    ref_text: str,
    gen_text: str,
    *,
    ref_sample_rate: int = TARGET_SAMPLE_RATE,
    nfe_step: int = NFE_STEP,
    cfg_strength: float = CFG_STRENGTH,
    sway_sampling_coef: float = SWAY_SAMPLING_COEF,
    speed: float = 1.0,
    target_rms: float = TARGET_RMS,
    device=None,
) -> np.ndarray:
    """Voice-clone ``gen_text`` from ``ref_audio``/``ref_text``; returns float32."""
    import torch

    patch_torchaudio()
    device = device or next(model.parameters()).device
    audio = _as_mono(ref_audio, ref_sample_rate, device)

    rms = audio.pow(2).mean().sqrt()
    quiet = bool(rms < target_rms)
    if quiet:
        audio = audio * (target_rms / rms.clamp(min=1e-8))

    ref_text = str(ref_text).strip()
    if not ref_text:
        raise ValueError("reference transcript is empty")
    if not ref_text.endswith((".", "!", "?")):
        ref_text += "."

    ref_seconds = audio.shape[-1] / TARGET_SAMPLE_RATE
    max_chars = max(40, int(len(ref_text.encode("utf-8")) / max(ref_seconds, 0.5) * (22 - ref_seconds) * speed))
    chunks = chunk_text(gen_text, max_chars=max_chars)

    ref_len = audio.shape[-1] // HOP_LENGTH
    ref_bytes = max(1, len(ref_text.encode("utf-8")))
    waves: list[np.ndarray] = []

    for chunk in chunks:
        local_speed = speed * 0.3 if len(chunk.encode("utf-8")) < 10 else speed
        duration = ref_len + int(ref_len / ref_bytes * len(chunk.encode("utf-8")) / local_speed)
        text = to_model_text(ref_text + " " + chunk)

        with torch.inference_mode():
            generated, _ = model.sample(
                cond=audio,
                text=[text],
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            mel = generated.to(torch.float32)[:, ref_len:, :].permute(0, 2, 1)
            wave = vocoder.decode(mel)
            if quiet:
                wave = wave * (rms / target_rms)
            waves.append(wave.squeeze().detach().cpu().numpy().astype(np.float32))
        del generated, mel

    return crossfade(waves, sample_rate=TARGET_SAMPLE_RATE)


def load_reference(paths: Paths):
    """Read ``ref_audio.wav`` + ``ref_text.txt`` produced by the data stage."""
    import torchaudio

    patch_torchaudio()
    if not paths.ref_audio.is_file() or not paths.ref_text_file.is_file():
        raise FileNotFoundError(f"{paths.ref_audio} / {paths.ref_text_file} missing - run `sauti-sheng data`")
    audio, sample_rate = torchaudio.load(str(paths.ref_audio))
    return audio, int(sample_rate), paths.ref_text_file.read_text(encoding="utf-8").strip()


def write_wav(path: Path, wave: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wave, dtype=np.float32), int(sample_rate), subtype="PCM_16")
    return path
