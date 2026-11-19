"""Collation identical to ``finetuning/dataset.py`` in QwenLM/Qwen3-TTS.

The layout is: one interleaved sequence of two channels. Channel 0 is text
(``[prefix(3)] | pad(4) | bos(1) | text | eos(1) | pad(codec_len)``), channel 1 is
codec (``pad(3) | nothink, think_bos, think_eos, <speaker slot>, pad | pad... |
bos(1) | audio_codes_0 | eos``). Labels are the codec-0 stream shifted by one.

Torch-only and config-driven (the 1.7B and 0.6B have different special-token ids
- ``codec_nothink_id`` is 4203 on 1.7B and 2155 on 0.6B), so it can be tested
without the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

SPEAKER_ROW = 6  # position whose codec embedding is replaced by the speaker vector


@dataclass(frozen=True)
class SeqIds:
    tts_pad_token_id: int
    tts_bos_token_id: int
    tts_eos_token_id: int
    codec_nothink_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    codec_pad_id: int
    codec_bos_id: int
    codec_eos_token_id: int
    num_code_groups: int = 16

    @classmethod
    def from_config(cls, config: Any) -> "SeqIds":
        """Accept either the Qwen3TTS config object or its ``config.json`` dict."""
        talker = config["talker_config"] if isinstance(config, dict) else config.talker_config
        top = config if isinstance(config, dict) else config
        get = (lambda k: top[k]) if isinstance(top, dict) else (lambda k: getattr(top, k))
        return cls(
            tts_pad_token_id=int(get("tts_pad_token_id")),
            tts_bos_token_id=int(get("tts_bos_token_id")),
            tts_eos_token_id=int(get("tts_eos_token_id")),
            codec_nothink_id=int(talker["codec_nothink_id"] if isinstance(talker, dict) else talker.codec_nothink_id),
            codec_think_bos_id=int(talker["codec_think_bos_id"] if isinstance(talker, dict) else talker.codec_think_bos_id),
            codec_think_eos_id=int(talker["codec_think_eos_id"] if isinstance(talker, dict) else talker.codec_think_eos_id),
            codec_pad_id=int(talker["codec_pad_id"] if isinstance(talker, dict) else talker.codec_pad_id),
            codec_bos_id=int(talker["codec_bos_id"] if isinstance(talker, dict) else talker.codec_bos_id),
            codec_eos_token_id=int(
                talker["codec_eos_token_id"] if isinstance(talker, dict) else talker.codec_eos_token_id
            ),
            num_code_groups=int(
                talker["num_code_groups"] if isinstance(talker, dict) else talker.num_code_groups
            ),
        )


def build_batch(items: Sequence[dict], ids: SeqIds) -> dict[str, torch.Tensor]:
    """``items``: dicts with ``text_ids`` (1,T) long, ``audio_codes`` (T,16), ``ref_mel`` (1,Tm,128)."""
    lengths = [item["text_ids"].shape[1] + item["audio_codes"].shape[0] for item in items]
    b = len(items)
    t = max(lengths) + 8

    input_ids = torch.zeros((b, t, 2), dtype=torch.long)
    codec_ids = torch.zeros((b, t, ids.num_code_groups), dtype=torch.long)
    text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_mask = torch.zeros((b, t), dtype=torch.bool)
    attention_mask = torch.zeros((b, t), dtype=torch.long)
    codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

    for i, item in enumerate(items):
        text_ids = item["text_ids"]
        audio_codes = item["audio_codes"]
        audio_codec_0 = audio_codes[:, 0]
        text_ids_len = text_ids.shape[1]
        codec_ids_len = audio_codec_0.shape[0]

        # ---- text channel
        input_ids[i, :3, 0] = text_ids[0, :3]
        input_ids[i, 3:7, 0] = ids.tts_pad_token_id
        input_ids[i, 7, 0] = ids.tts_bos_token_id
        input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
        input_ids[i, 8 + text_ids_len - 3, 0] = ids.tts_eos_token_id
        input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = ids.tts_pad_token_id
        text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

        # ---- codec channel
        input_ids[i, 3:8, 1] = torch.tensor(
            [
                ids.codec_nothink_id,
                ids.codec_think_bos_id,
                ids.codec_think_eos_id,
                0,  # replaced by the speaker embedding
                ids.codec_pad_id,
            ]
        )
        input_ids[i, 8 : 8 + text_ids_len - 3, 1] = ids.codec_pad_id
        input_ids[i, 8 + text_ids_len - 3, 1] = ids.codec_pad_id
        input_ids[i, 8 + text_ids_len - 2, 1] = ids.codec_bos_id
        input_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
        input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = ids.codec_eos_token_id

        codec_0_labels[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
        codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = ids.codec_eos_token_id

        codec_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :] = audio_codes

        codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
        codec_embedding_mask[i, SPEAKER_ROW] = False  # speaker slot

        codec_mask[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = True
        attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

    return {
        "input_ids": input_ids,
        "ref_mels": torch.cat([item["ref_mel"] for item in items], dim=0),
        "attention_mask": attention_mask,
        "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
        "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
        "codec_0_labels": codec_0_labels,
        "codec_ids": codec_ids,
        "codec_mask": codec_mask,
    }