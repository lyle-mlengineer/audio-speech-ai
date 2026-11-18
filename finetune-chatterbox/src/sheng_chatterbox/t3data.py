"""Turn (wav, text) pairs into the tensors Chatterbox's T3 expects.

One training example is exactly what ``ChatterboxMultilingualTTS.generate``
consumes at inference time:

* ``text_tokens``   - grapheme tokens, wrapped in ``[START]`` / ``[STOP]``
* ``speech_tokens`` - S3 tokens of the clip, wrapped in start/stop speech tokens
* ``speaker_emb``   - 256-d voice-encoder embedding (the "who")
* ``cond_tokens``   - S3 tokens of the first ``cond_seconds`` of the clip
                      (the "how", ``speech_cond_prompt_len`` = 150 tokens)
* ``emotion_adv``   - exaggeration scalar, 0.5 = neutral (what training uses)

Featurisation is the expensive part (s3gen's tokenizer runs a full mel/quantiser
forward per clip), so results are cached as ``.npz`` next to the checkpoints.
Cache keys include the transcript and the featuriser settings, so editing text or
changing ``cond_seconds`` invalidates automatically.

Torch is imported inside the functions: ``sheng-chatterbox status`` must keep
working on a VM where torch is not installed yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import manifest as M

# S3 tokenizer input rate (chatterbox.models.s3tokenizer.S3_SR).
S3_SR = 16_000


@dataclass
class FeaturizeConfig:
    language_id: str = "sw"
    model_variant: str = "mtl"
    max_text_len: int = 512
    max_speech_len: int = 1024
    cond_seconds: float = 6.0
    emotion_adv: float = 0.5
    cache_dir: Path | None = None

    def signature(self) -> str:
        """Everything that changes the cached tensors (not the paths)."""
        blob = json.dumps(
            {
                "language_id": self.language_id,
                "model_variant": self.model_variant,
                "max_text_len": self.max_text_len,
                "max_speech_len": self.max_speech_len,
                "cond_seconds": round(self.cond_seconds, 3),
                "emotion_adv": round(self.emotion_adv, 3),
            },
            sort_keys=True,
        )
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _load_wav16k(path: Path):
    import librosa

    wav, _sr = librosa.load(str(path), sr=S3_SR, mono=True)
    return wav


class Featurizer:
    """Stateless per-clip tensor builder (holds the frozen encoders)."""

    def __init__(self, model, cfg: FeaturizeConfig, device: str = "cuda"):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.hp = model.t3.hp
        self._sig = cfg.signature()
        if cfg.cache_dir is not None:
            cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ text
    def text_tokens(self, text: str):
        """``[START] text [STOP]``, clipped to ``max_text_len``."""
        import torch

        if self.cfg.model_variant == "mtl":
            toks = self.model.tokenizer.text_to_tokens(text, language_id=self.cfg.language_id)
        else:
            toks = self.model.tokenizer.text_to_tokens(text)
        toks = toks.squeeze(0).long()
        room = max(1, self.cfg.max_text_len - 2)
        if toks.numel() > room:
            toks = toks[:room]
        special = torch.tensor(
            [self.hp.start_text_token, self.hp.stop_text_token], dtype=torch.long
        )
        return torch.cat([special[:1], toks, special[1:]])

    # ---------------------------------------------------------------- speech
    def speech_tokens(self, wav):
        """S3 tokens of the whole clip, clipped to ``max_speech_len``."""
        import torch

        ids, lens = self.model.s3gen.tokenizer.forward([wav])
        toks = ids[0, : int(lens[0])].long() if ids is not None else torch.zeros(0, dtype=torch.long)
        room = max(1, self.cfg.max_speech_len - 2)
        if toks.numel() > room:
            toks = toks[:room]
        if toks.numel() == 0:
            raise ValueError("empty speech token sequence (clip too short or silent)")
        special = torch.tensor(
            [self.hp.start_speech_token, self.hp.stop_speech_token], dtype=torch.long
        )
        return torch.cat([special[:1], toks, special[1:]])

    def cond_tokens(self, wav):
        """S3 tokens of the first ``cond_seconds`` - the in-context voice prompt."""
        import torch

        prompt_len = int(self.hp.speech_cond_prompt_len)
        seg = wav[: int(self.cfg.cond_seconds * S3_SR)]
        if seg.size == 0:
            return torch.zeros(prompt_len, dtype=torch.long)
        ids, _lens = self.model.s3gen.tokenizer.forward([seg], max_len=prompt_len)
        if ids is None or ids.numel() == 0:
            return torch.zeros(prompt_len, dtype=torch.long)
        toks = ids[0].long()
        if toks.numel() > prompt_len:
            toks = toks[:prompt_len]
        elif toks.numel() < prompt_len:
            toks = torch.cat([toks, torch.zeros(prompt_len - toks.numel(), dtype=torch.long)])
        return toks

    def speaker_embedding(self, wav):
        import torch

        emb = self.model.ve.embeds_from_wavs([wav], sample_rate=S3_SR)
        return torch.from_numpy(emb).float().reshape(-1)

    # ------------------------------------------------------------------ clip
    def __call__(self, audio_path: Path, text: str) -> dict[str, Any]:
        import torch

        wav = _load_wav16k(audio_path)
        return {
            "text_tokens": self.text_tokens(text),
            "speech_tokens": self.speech_tokens(wav),
            "t3_cond_speaker_emb": self.speaker_embedding(wav),
            "t3_cond_prompt_speech_tokens": self.cond_tokens(wav),
            "t3_cond_emotion_adv": torch.tensor(float(self.cfg.emotion_adv), dtype=torch.float32),
        }

    # ---------------------------------------------------------------- caching
    def cache_path(self, audio_path: Path, text: str, mtime: float) -> Path | None:
        if self.cfg.cache_dir is None:
            return None
        key = hashlib.sha1(
            f"{Path(audio_path).resolve()}|{mtime:.5f}|{text}|{self._sig}".encode("utf-8")
        ).hexdigest()[:16]
        return Path(self.cfg.cache_dir) / f"{Path(audio_path).stem}_{key}.npz"

    def load_cached(self, path: Path) -> dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        try:
            import numpy as np
            import torch

            with np.load(path) as z:
                return {
                    "text_tokens": torch.from_numpy(z["text_tokens"].astype("int64")),
                    "speech_tokens": torch.from_numpy(z["speech_tokens"].astype("int64")),
                    "t3_cond_speaker_emb": torch.from_numpy(z["t3_cond_speaker_emb"].astype("float32")),
                    "t3_cond_prompt_speech_tokens": torch.from_numpy(
                        z["t3_cond_prompt_speech_tokens"].astype("int64")
                    ),
                    "t3_cond_emotion_adv": torch.tensor(
                        float(z["t3_cond_emotion_adv"]), dtype=torch.float32
                    ),
                }
        except Exception:  # noqa: BLE001 - a corrupt cache entry is just a miss
            return None

    def save_cached(self, path: Path, item: dict[str, Any]) -> None:
        if path is None:
            return
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            text_tokens=item["text_tokens"].numpy(),
            speech_tokens=item["speech_tokens"].numpy(),
            t3_cond_speaker_emb=item["t3_cond_speaker_emb"].numpy(),
            t3_cond_prompt_speech_tokens=item["t3_cond_prompt_speech_tokens"].numpy(),
            t3_cond_emotion_adv=np.float32(float(item["t3_cond_emotion_adv"])),
        )
        tmp.replace(path)


def resolve_audio(manifest_path: Path, audio: str) -> Path:
    p = Path(audio)
    return p if p.is_absolute() else (Path(manifest_path).parent / p)


class T3Dataset:
    """Map-style dataset over a manifest; item is a tensor dict (or None on error)."""

    def __init__(self, manifest_path: Path, featurizer: Featurizer, *, limit: int | None = None):
        self.manifest_path = Path(manifest_path)
        self.featurizer = featurizer
        self.rows = M.read_manifest(self.manifest_path)
        if limit:
            self.rows = self.rows[: int(limit)]
        self.failures: list[str] = []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        row = self.rows[idx]
        audio = resolve_audio(self.manifest_path, row["audio"])
        text = (row.get("text") or "").strip()
        if not audio.exists() or not text:
            self.failures.append(f"{audio}: missing audio or text")
            return None
        try:
            mtime = audio.stat().st_mtime
            cache = self.featurizer.cache_path(audio, text, mtime)
            item = self.featurizer.load_cached(cache) if cache else None
            if item is None:
                item = self.featurizer(audio, text)
                if cache:
                    self.featurizer.save_cached(cache, item)
            return item
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
            self.failures.append(f"{audio}: {exc}")
            return None


@dataclass
class SpeechCollator:
    """Pad to the longest item and stack the per-clip conditionals."""

    text_pad: int = 0
    speech_pad: int = 0

    def __call__(self, features: Iterable[dict[str, Any] | None]) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        feats = [f for f in features if f is not None]
        if not feats:
            return {}

        max_text = max(int(f["text_tokens"].numel()) for f in feats)
        max_speech = max(int(f["speech_tokens"].numel()) for f in feats)

        text = torch.stack([F.pad(f["text_tokens"], (0, max_text - f["text_tokens"].numel()), value=self.text_pad) for f in feats])
        speech = torch.stack(
            [F.pad(f["speech_tokens"], (0, max_speech - f["speech_tokens"].numel()), value=self.speech_pad) for f in feats]
        )
        return {
            "text_tokens": text,
            "text_token_lens": torch.tensor([int(f["text_tokens"].numel()) for f in feats], dtype=torch.long),
            "speech_tokens": speech,
            "speech_token_lens": torch.tensor([int(f["speech_tokens"].numel()) for f in feats], dtype=torch.long),
            "t3_cond_speaker_emb": torch.stack([f["t3_cond_speaker_emb"] for f in feats]),
            "t3_cond_prompt_speech_tokens": torch.stack(
                [f["t3_cond_prompt_speech_tokens"] for f in feats]
            ),
            "t3_cond_emotion_adv": torch.stack(
                [f["t3_cond_emotion_adv"] for f in feats]
            ).view(len(feats), 1, 1),
        }


def endless(loader):
    """Cycle a DataLoader forever, so the loop is step-based, not epoch-based."""
    while True:
        for batch in loader:
            if batch:
                yield batch
