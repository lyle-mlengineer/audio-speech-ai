"""The 0.6B regression test: 2048-dim text embeddings must be projected to 1024
before they are added to the codec embeddings (issues #198/#335)."""

from __future__ import annotations

import torch
from torch import nn

from sheng_qwentts.embeds import assemble_inputs_embeds


class FakeResize(nn.Module):
    """Same shape as Qwen3TTSTalkerResizeMLP, bias included on purpose."""

    def __init__(self, inp: int, out: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(inp, inp, bias=True)
        self.linear_fc2 = nn.Linear(inp, out, bias=True)

    def forward(self, x):
        return self.linear_fc2(torch.relu(self.linear_fc1(x)))


class FakeTalker(nn.Module):
    def __init__(self, text_hidden: int = 2048, hidden: int = 1024, groups: int = 16):
        super().__init__()
        self.model = nn.Module()
        self.model.text_embedding = nn.Embedding(64, text_hidden)
        self.model.codec_embedding = nn.Embedding(64, hidden)
        self.text_projection = FakeResize(text_hidden, hidden)
        self.code_predictor = nn.Module()
        self.code_predictor._embeddings = nn.ModuleList([nn.Embedding(64, hidden) for _ in range(groups - 1)])

    def _get(self):
        return list(self.code_predictor._embeddings)


class FakeCodePredictor(nn.Module):
    pass


def _talker_with_predictor(**kwargs) -> FakeTalker:
    talker = FakeTalker(**kwargs)
    talker.code_predictor.get_input_embeddings = talker._get  # type: ignore[attr-defined]
    return talker


def _inputs(b: int = 1, t: int = 12, groups: int = 16):
    g = torch.Generator().manual_seed(0)
    text_ids = torch.randint(0, 64, (b, t), generator=g)
    codec_ids = torch.randint(0, 64, (b, t, groups), generator=g)
    text_mask = torch.ones(b, t, 1, dtype=torch.bool)
    text_mask[:, 0] = False  # a padded position
    codec_embedding_mask = torch.ones(b, t, 1, dtype=torch.bool)
    codec_embedding_mask[:, 0] = False
    codec_mask = torch.ones(b, t, dtype=torch.bool)  # (b, t): the trainer's codec_mask
    codec_mask[:, 0] = False
    return text_ids, codec_ids, text_mask, codec_embedding_mask, codec_mask


def test_projects_to_hidden_size():
    talker = _talker_with_predictor()
    text_ids, codec_ids, text_mask, codec_embedding_mask, codec_mask = _inputs()
    embeds = assemble_inputs_embeds(
        talker=talker,
        input_text_ids=text_ids,
        input_codec_ids=torch.zeros_like(text_ids),
        codec_ids=codec_ids,
        text_embedding_mask=text_mask,
        codec_embedding_mask=codec_embedding_mask,
        codec_mask=codec_mask,
        speaker_embedding=torch.zeros(1, 1024),
    )
    assert embeds.shape == (1, 12, 1024)


def test_raw_text_embedding_would_have_been_the_wrong_width():
    """Documents the upstream bug: without the projection the add cannot work."""
    talker = _talker_with_predictor()
    text_ids, codec_ids, _, _, _ = _inputs()
    raw_text = talker.model.text_embedding(text_ids)
    raw_codec = talker.model.codec_embedding(torch.zeros_like(text_ids))
    assert raw_text.shape[-1] == 2048
    assert raw_codec.shape[-1] == 1024
    try:
        _ = raw_text + raw_codec
    except RuntimeError:
        pass
    else:  # pragma: no cover - torch version without the shape check
        raise AssertionError("expected a broadcast failure for the unprojected add")


def test_mask_is_reapplied_after_the_biased_projection():
    talker = _talker_with_predictor()
    text_ids, codec_ids, text_mask, codec_embedding_mask, codec_mask = _inputs()
    # make the projection itself non-zero everywhere so bias leakage is visible
    with torch.no_grad():
        talker.text_projection.linear_fc2.bias.fill_(1.0)
        talker.text_projection.linear_fc1.bias.fill_(1.0)
    embeds = assemble_inputs_embeds(
        talker=talker,
        input_text_ids=text_ids,
        input_codec_ids=torch.zeros_like(text_ids),
        codec_ids=torch.zeros_like(codec_ids),  # all sub-codebook embeddings of token 0
        text_embedding_mask=text_mask,
        codec_embedding_mask=codec_embedding_mask,
        codec_mask=codec_mask,
        speaker_embedding=torch.zeros(1, 1024),
    )
    # everything that is masked on both channels must be exactly zero
    assert torch.equal(embeds[0, 0], torch.zeros(1024))


def test_speaker_embedding_lands_in_row_6():
    talker = _talker_with_predictor()
    text_ids, codec_ids, text_mask, codec_embedding_mask, codec_mask = _inputs()
    speaker = torch.zeros(1, 1024)
    speaker[0, 7] = 5.0
    embeds = assemble_inputs_embeds(
        talker=talker,
        input_text_ids=text_ids,
        input_codec_ids=torch.zeros_like(text_ids),
        codec_ids=torch.zeros_like(codec_ids),
        text_embedding_mask=torch.zeros(1, 12, 1, dtype=torch.bool),
        codec_embedding_mask=torch.ones(1, 12, 1, dtype=torch.bool),
        codec_mask=torch.zeros(1, 12, dtype=torch.bool),  # isolate the speaker row
        speaker_embedding=speaker,
    )
    assert embeds[0, 6, 7].item() == 5.0