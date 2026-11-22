from __future__ import annotations

import pytest

from sauti_sheng.checkpoint import (
    EMA_KEY,
    ONLINE_KEY,
    describe,
    export_state,
    has_ema,
    looks_like_training_checkpoint,
    prepare_for_load,
    select_state_dict,
    state_nbytes,
    state_tensor_count,
    strip_to_ema,
    strip_to_online,
    text_embed_rows,
    text_vocab_size,
    top_level_keys,
    vocab_size_from_rows,
)


class FakeTensor:
    def __init__(self, shape, dtype_bytes=4):
        self.shape = tuple(shape)
        self._dtype_bytes = dtype_bytes

    def numel(self):
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    def element_size(self):
        return self._dtype_bytes


def _ema():
    return {
        "ema_model.transformer.text_embed.text_embed.weight": FakeTensor((2546, 512)),
        "ema_model.transformer.x": FakeTensor((2, 3)),
        "initted": True,
        "step": 15350,
    }


def _release():
    return {
        "model_state_dict": {"transformer.x": FakeTensor((2, 3))},
        "ema_model_state_dict": _ema(),
        "optimizer_state_dict": {"state": {}},
        "scheduler_state_dict": {"last_epoch": 15350},
        "update": 15350,
    }


def test_release_is_detected_as_training_checkpoint():
    assert looks_like_training_checkpoint(_release())
    assert not looks_like_training_checkpoint({EMA_KEY: _ema()})


def test_strip_to_ema_drops_resume_keys():
    stripped = strip_to_ema(_release())
    assert list(stripped) == [EMA_KEY]
    assert not looks_like_training_checkpoint(stripped)


def test_strip_to_ema_requires_ema():
    with pytest.raises(KeyError, match=EMA_KEY):
        strip_to_ema({"model_state_dict": {}})


def test_select_online_prefers_model_state_dict():
    state = select_state_dict(_release(), use_ema=False)
    assert set(state) == {"transformer.x"}


def test_select_ema_strips_prefix_and_metadata():
    state = select_state_dict(_release(), use_ema=True)
    assert "transformer.text_embed.text_embed.weight" in state
    assert all(not key.startswith("ema_model.") for key in state)
    assert "initted" not in state and "step" not in state


def test_select_falls_back_between_the_two():
    only_ema = {EMA_KEY: _ema()}
    assert "transformer.x" in select_state_dict(only_ema, use_ema=False)
    only_online = {ONLINE_KEY: {"transformer.x": FakeTensor((1,))}}
    assert "transformer.x" in select_state_dict(only_online, use_ema=True)


def test_select_rejects_empty():
    with pytest.raises(KeyError):
        select_state_dict({"update": 3})


def test_export_state_shapes():
    online = export_state(_release(), use_ema=False)
    assert list(online) == [ONLINE_KEY]
    assert all(not key.startswith("ema_model.") for key in online[ONLINE_KEY])

    ema = export_state(_release(), use_ema=True)
    assert list(ema) == [EMA_KEY]
    assert any(key.startswith("ema_model.") for key in ema[EMA_KEY])


def test_prepare_for_load_drops_stale_buffers():
    state = {
        "mel_spec.mel_stft.mel_scale.fb": FakeTensor((1,)),
        "mel_spec.mel_stft.spectrogram.window": FakeTensor((1,)),
        "transformer.x": FakeTensor((1,)),
    }
    assert list(prepare_for_load(state)) == ["transformer.x"]


def test_strip_to_online_is_a_single_key():
    assert list(strip_to_online(_release())) == [ONLINE_KEY]


def test_text_vocab_size_is_the_row_count_not_the_vocab_size():
    assert text_vocab_size(_ema()) == 2546
    assert text_vocab_size({"transformer.x": FakeTensor((3,))}) is None


def test_text_embed_is_one_row_larger_than_the_vocab():
    # F5: TextEmbedding -> nn.Embedding(text_num_embeds + 1, text_dim)
    assert text_embed_rows(2545) == 2546
    assert vocab_size_from_rows(2546) == 2545
    assert vocab_size_from_rows(text_embed_rows(2545)) == 2545


def test_describe_summarises():
    info = describe(_release())
    assert info["training_checkpoint"] is True
    assert info["text_embed_rows"] == 2546
    assert info["text_vocab_size"] == 2545
    assert info["ema_tensors"] == 2
    assert info["update"] == 15350
    assert info["ema_bytes"] == (2546 * 512 + 6) * 4


def test_state_helpers():
    assert state_tensor_count({"a": FakeTensor((1,)), "b": 1}) == 1
    assert state_nbytes({"a": FakeTensor((4,), dtype_bytes=2)}) == 8
    assert top_level_keys({"b": 1, "a": 2}) == ["a", "b"]
    assert has_ema({EMA_KEY: {"x": 1}})
