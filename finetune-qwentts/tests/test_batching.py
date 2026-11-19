from __future__ import annotations

import torch
import pytest

from sheng_qwentts.batching import SeqIds, build_batch

# Snapshot of the 0.6B config.json values (they differ from 1.7B: codec ids).
IDS = SeqIds(
    tts_pad_token_id=151671,
    tts_bos_token_id=151672,
    tts_eos_token_id=151673,
    codec_nothink_id=2155,
    codec_think_bos_id=2156,
    codec_think_eos_id=2157,
    codec_pad_id=2148,
    codec_bos_id=2149,
    codec_eos_token_id=2150,
    num_code_groups=16,
)


def _item(text_len: int = 20, codec_len: int = 40, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "text_ids": torch.randint(100, 150000, (1, text_len), generator=g),
        "audio_codes": torch.randint(0, 3072, (codec_len, 16), generator=g),
        "ref_mel": torch.randn(1, 30, 128),
    }


def test_shapes_and_dtypes():
    batch = build_batch([_item(), _item(35, 50)], IDS)
    b, t = 2, max(20 + 40, 35 + 50) + 8
    assert batch["input_ids"].shape == (b, t, 2)
    assert batch["codec_ids"].shape == (b, t, 16)
    assert batch["text_embedding_mask"].shape == (b, t, 1)
    assert batch["codec_embedding_mask"].shape == (b, t, 1)
    assert batch["codec_mask"].shape == (b, t)
    assert batch["attention_mask"].shape == (b, t)
    assert batch["codec_0_labels"].shape == (b, t)
    assert batch["ref_mels"].shape == (b, 30, 128)
    assert batch["input_ids"].dtype == torch.long


def test_text_channel_layout():
    item = _item(text_len=20)
    batch = build_batch([item], IDS)
    ids = batch["input_ids"][0, :, 0]
    text = item["text_ids"][0]
    assert torch.equal(ids[8 : 8 + 17], text[3:])  # text body (20-3=17)
    assert ids[7].item() == IDS.tts_bos_token_id
    assert ids[8 + 17].item() == IDS.tts_eos_token_id
    assert ids[3:7].tolist() == [IDS.tts_pad_token_id] * 4


def test_codec_channel_and_labels():
    item = _item(codec_len=40)
    batch = build_batch([item], IDS)
    ids1 = batch["input_ids"][0, :, 1]
    codes0 = item["audio_codes"][:, 0]
    text_len = item["text_ids"].shape[1]
    start = 8 + text_len - 1
    assert torch.equal(ids1[start : start + 40], codes0)
    assert ids1[start + 40].item() == IDS.codec_eos_token_id
    # binary cross-check of the layout upstream relies on:
    assert torch.equal(batch["codec_0_labels"][0, start : start + 40], codes0)
    assert batch["codec_0_labels"][0, start + 40].item() == IDS.codec_eos_token_id
    assert batch["codec_0_labels"][0, 0].item() == -100


def test_speaker_slot_is_masked_out_of_the_codec_embedding():
    batch = build_batch([_item()], IDS)
    row6 = batch["codec_embedding_mask"][0, 6]
    assert not bool(row6.item())
    assert batch["codec_embedding_mask"][0, 5].bool()
    assert batch["codec_embedding_mask"][0, 7].bool()


def test_masks_cover_attention_length():
    batch = build_batch([_item(text_len=20, codec_len=40)], IDS)
    seq_len = 8 + 20 + 40
    assert batch["attention_mask"][0, :seq_len].all()
    assert not batch["attention_mask"][0, seq_len:].any()
    assert batch["text_embedding_mask"][0, :seq_len].all()


def test_seqids_from_config_dict():
    config = {
        "tts_pad_token_id": 1, "tts_bos_token_id": 2, "tts_eos_token_id": 3,
        "talker_config": {
            "codec_nothink_id": 11, "codec_think_bos_id": 12, "codec_think_eos_id": 13,
            "codec_pad_id": 14, "codec_bos_id": 15, "codec_eos_token_id": 16,
            "num_code_groups": 16,
        },
    }
    ids = SeqIds.from_config(config)
    assert ids.codec_nothink_id == 11 and ids.tts_pad_token_id == 1
    assert ids.num_code_groups == 16


def test_variable_lengths_pack_into_one_matrix():
    batch = build_batch([_item(10, 30), _item(40, 60)], IDS)
    assert batch["attention_mask"].sum(1).tolist() == [8 + 10 + 30, 8 + 40 + 60]