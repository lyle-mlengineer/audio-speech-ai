"""The embedding->input assembly for SFT, shaped by the upstream trainer but
kept dependency-free (torch only) so the 0.6B fix is unit-testable.

Why the fix exists (issues #198/#335 in QwenLM/Qwen3-TTS): on the 0.6B model
``text_hidden_size`` (2048) != ``hidden_size`` (1024). The upstream
``sft_12hz.py`` adds the raw text embedding to the codec embedding element-wise,
which crashes on 0.6B and would silently corrupt on 1.7B. The inference path
already routes the text embedding through ``talker.text_projection``; fine-tuning
must do the same. ``text_projection`` has bias, so the mask is re-applied after
projecting (padded positions would otherwise pick up the bias).
"""

from __future__ import annotations

import torch


def assemble_inputs_embeds(
    *,
    talker,
    input_text_ids: torch.Tensor,
    input_codec_ids: torch.Tensor,
    codec_ids: torch.Tensor,
    text_embedding_mask: torch.Tensor,
    codec_embedding_mask: torch.Tensor,
    codec_mask: torch.Tensor,
    speaker_embedding: torch.Tensor,
    speaker_embedding_row: int = 6,
) -> torch.Tensor:
    """Same layout as ``Qwen3TTSModel`` inference: text (projected) + codec_0 +
    the speaker embedding injected at ``codec_embedding_mask``-off position 6 +
    every sub-codebook embedding.

    ``talker`` duck-types: ``model.text_embedding``, ``text_projection``,
    ``model.codec_embedding``, ``code_predictor.get_input_embeddings()``.
    """
    text_embedding = talker.model.text_embedding(input_text_ids.to(torch.long))
    # 0.6B fix: project 2048 -> 1024 before adding; re-mask after (bias != 0).
    text_embedding = talker.text_projection(text_embedding)
    text_embedding = text_embedding * text_embedding_mask.to(text_embedding.dtype)

    codec_embedding = talker.model.codec_embedding(input_codec_ids.to(torch.long))
    codec_embedding = codec_embedding * codec_embedding_mask.to(codec_embedding.dtype)
    codec_embedding[:, speaker_embedding_row, :] = speaker_embedding

    embeds = text_embedding + codec_embedding

    num_groups = codec_ids.shape[-1]
    for i in range(1, num_groups):
        sub_embedding = talker.code_predictor.get_input_embeddings()[i - 1](
            codec_ids[:, :, i].to(torch.long)
        )
        embeds = embeds + sub_embedding * codec_mask.unsqueeze(-1).to(sub_embedding.dtype)
    return embeds