"""Featurisation, caching and collation (no GPU, no chatterbox install needed).

A fake model stands in for Chatterbox: the real one is only used for
``tokenizer.text_to_tokens``, ``s3gen.tokenizer.forward`` and
``ve.embeds_from_wavs``, so those three are all a fake needs.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from sheng_chatterbox import t3data  # noqa: E402


class FakeS3Tokenizer:
    def __init__(self, tokens_per_second: int = 25):
        self.tps = tokens_per_second
        self.calls: list[int] = []

    def forward(self, wavs, max_len=None):
        wav = np.asarray(wavs[0])
        self.calls.append(len(wav))
        n = max(1, len(wav) // (16_000 // self.tps))
        if max_len is not None:
            n = min(n, max_len)
        ids = torch.arange(n).unsqueeze(0).long()
        lens = torch.tensor([n], dtype=torch.long)
        return ids, lens


class FakeTextTokenizer:
    def __init__(self):
        self.seen: list[tuple[str, str | None]] = []

    def text_to_tokens(self, text, language_id=None):
        self.seen.append((text, language_id))
        return torch.arange(1, len(text) + 1).unsqueeze(0).long()


class FakeVE:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embeds_from_wavs(self, wavs, sample_rate):
        return np.zeros((len(wavs), self.dim), dtype="float32")


def make_model(**hp_overrides):
    hp = types.SimpleNamespace(
        start_text_token=255,
        stop_text_token=0,
        start_speech_token=6561,
        stop_speech_token=6562,
        speech_cond_prompt_len=150,
        max_text_tokens=2048,
        max_speech_tokens=4096,
    )
    for key, value in hp_overrides.items():
        setattr(hp, key, value)
    return types.SimpleNamespace(
        t3=types.SimpleNamespace(hp=hp),
        tokenizer=FakeTextTokenizer(),
        s3gen=types.SimpleNamespace(tokenizer=FakeS3Tokenizer()),
        ve=FakeVE(),
    )


@pytest.fixture()
def fake_wav(monkeypatch):
    """Pretend every path decodes to 6 s of 16 kHz audio."""
    monkeypatch.setattr(t3data, "_load_wav16k", lambda path: np.zeros(6 * 16_000, dtype="float32"))


def test_text_tokens_are_wrapped_in_bot_eot():
    model = make_model()
    featurizer = t3data.Featurizer(model, t3data.FeaturizeConfig(), device="cpu")
    toks = featurizer.text_tokens("Sasa wewe")

    assert int(toks[0]) == 255 and int(toks[-1]) == 0
    assert toks.numel() == len("Sasa wewe") + 2
    # the multilingual tokenizer must be told the language
    assert model.tokenizer.seen == [("Sasa wewe", "sw")]


def test_text_is_clipped_to_max_text_len():
    model = make_model()
    cfg = t3data.FeaturizeConfig(max_text_len=10)
    toks = t3data.Featurizer(model, cfg, device="cpu").text_tokens("x" * 500)
    assert toks.numel() == 10


def test_english_variant_does_not_pass_a_language():
    model = make_model()
    cfg = t3data.FeaturizeConfig(model_variant="english")
    t3data.Featurizer(model, cfg, device="cpu").text_tokens("hello")
    assert model.tokenizer.seen == [("hello", None)]


def test_speech_tokens_are_wrapped_and_clipped():
    model = make_model()
    cfg = t3data.FeaturizeConfig(max_speech_len=20)
    featurizer = t3data.Featurizer(model, cfg, device="cpu")
    wav = np.zeros(30 * 16_000, dtype="float32")

    toks = featurizer.speech_tokens(wav)
    assert int(toks[0]) == 6561 and int(toks[-1]) == 6562
    assert toks.numel() == 20  # 1024-cap equivalent: clipped, then wrapped


def test_empty_speech_is_an_error():
    model = make_model()
    model.s3gen.tokenizer.forward = lambda wavs, max_len=None: (
        torch.zeros(1, 0, dtype=torch.long),
        torch.tensor([0]),
    )
    featurizer = t3data.Featurizer(model, t3data.FeaturizeConfig(), device="cpu")
    with pytest.raises(ValueError, match="empty speech token"):
        featurizer.speech_tokens(np.zeros(1000, dtype="float32"))


def test_cond_tokens_are_padded_to_the_prompt_len():
    model = make_model(speech_cond_prompt_len=150)
    featurizer = t3data.Featurizer(model, t3data.FeaturizeConfig(cond_seconds=6.0), device="cpu")
    cond = featurizer.cond_tokens(np.zeros(6 * 16_000, dtype="float32"))
    assert cond.numel() == 150
    assert cond.dtype == torch.long


def test_cond_tokens_are_truncated_to_the_prompt_len():
    model = make_model(speech_cond_prompt_len=10)
    featurizer = t3data.Featurizer(model, t3data.FeaturizeConfig(cond_seconds=6.0), device="cpu")
    assert featurizer.cond_tokens(np.zeros(6 * 16_000, dtype="float32")).numel() == 10


def test_full_item_contract(fake_wav):
    model = make_model()
    featurizer = t3data.Featurizer(model, t3data.FeaturizeConfig(), device="cpu")
    item = featurizer(Path("a.wav"), "Niaje msee")

    assert set(item) == {
        "text_tokens",
        "speech_tokens",
        "t3_cond_speaker_emb",
        "t3_cond_prompt_speech_tokens",
        "t3_cond_emotion_adv",
    }
    assert item["t3_cond_speaker_emb"].shape == (256,)
    assert item["t3_cond_prompt_speech_tokens"].numel() == 150
    assert float(item["t3_cond_emotion_adv"]) == 0.5


def test_cache_roundtrip_and_invalidation(fake_wav, tmp_path):
    model = make_model()
    cfg = t3data.FeaturizeConfig(cache_dir=tmp_path / "cache")
    featurizer = t3data.Featurizer(model, cfg, device="cpu")
    wav_path = tmp_path / "a.wav"
    wav_path.write_bytes(b"x")

    item = featurizer(wav_path, "Sasa")
    cache = featurizer.cache_path(wav_path, "Sasa", 1.0)
    featurizer.save_cached(cache, item)

    back = featurizer.load_cached(cache)
    assert torch.equal(back["text_tokens"], item["text_tokens"])
    assert torch.equal(back["speech_tokens"], item["speech_tokens"])
    assert torch.allclose(back["t3_cond_speaker_emb"], item["t3_cond_speaker_emb"])

    # transcript / featuriser changes must produce a different cache key
    assert featurizer.cache_path(wav_path, "Sasa!", 1.0) != cache
    other = t3data.Featurizer(model, t3data.FeaturizeConfig(cache_dir=tmp_path / "cache", cond_seconds=3.0))
    assert other.cache_path(wav_path, "Sasa", 1.0) != cache


def test_collate_pads_to_the_batch_maximum(fake_wav):
    model = make_model()
    featurizer = t3data.Featurizer(model, t3data.FeaturizeConfig(), device="cpu")
    items = [featurizer(Path("a.wav"), "one"), featurizer(Path("b.wav"), "a much longer sentence")]

    batch = t3data.SpeechCollator()(items)
    assert batch["text_tokens"].shape[0] == 2
    assert batch["text_tokens"].shape[1] == int(batch["text_token_lens"].max())
    assert batch["speech_tokens"].shape[1] == int(batch["speech_token_lens"].max())
    assert batch["t3_cond_speaker_emb"].shape == (2, 256)
    assert batch["t3_cond_prompt_speech_tokens"].shape == (2, 150)
    assert batch["t3_cond_emotion_adv"].shape == (2, 1, 1)


def test_collate_drops_unusable_items(fake_wav):
    model = make_model()
    featurizer = t3data.Featurizer(model, t3data.FeaturizeConfig(), device="cpu")
    batch = t3data.SpeechCollator()([None, featurizer(Path("a.wav"), "one"), None])
    assert batch["text_tokens"].shape[0] == 1
    assert t3data.SpeechCollator()([]) == {}


def test_dataset_skips_missing_audio(tmp_path):
    from sheng_chatterbox import manifest as M

    path = tmp_path / "manifests" / "train.jsonl"
    good = tmp_path / "manifests" / "a.wav"
    good.write_bytes(b"x")
    M.write_manifest(
        path,
        [
            {"audio": "a.wav", "text": "ok"},
            {"audio": "gone.wav", "text": "ok"},
        ],
    )
    ds = t3data.T3Dataset(path, featurizer=object())
    assert len(ds) == 2
    assert ds[1] is None
    assert "gone.wav" in " ".join(ds.failures)


def test_endless_cycles_the_loader():
    assert list(zip(range(5), t3data.endless(iter([[1], [2]])))) == [(0, [1]), (1, [2]), (2, [1]), (3, [2]), (4, [1])]
