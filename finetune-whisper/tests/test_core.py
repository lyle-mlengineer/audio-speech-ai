"""Tests for config, normalisation, metrics and augmentation.

These run without torch/datasets so they are fast and CI-friendly:
    python -m pytest tests/ -q
    python tests/test_core.py          # no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_sheng.config import Config  # noqa: E402
from whisper_sheng.metrics import compute_metrics, english_token_fraction  # noqa: E402
from whisper_sheng.text_norm import normalize_text, tokens  # noqa: E402


# ------------------------------------------------------------------- config


def test_default_config_is_whisper_medium():
    cfg = Config()
    assert cfg.model.name == "openai/whisper-medium"
    assert cfg.model.language == "swahili"


def test_model_name_override_is_arbitrary():
    cfg = Config()
    cfg.apply_overrides({"model.name": "openai/whisper-large-v3"})
    assert cfg.model.name == "openai/whisper-large-v3"


def test_override_type_coercion():
    cfg = Config()
    cfg.apply_overrides(
        {
            "train.per_device_train_batch_size": "2",
            "train.learning_rate": "3e-5",
            "train.fp16": "false",
        }
    )
    assert cfg.train.per_device_train_batch_size == 2
    assert abs(cfg.train.learning_rate - 3e-5) < 1e-12
    assert cfg.train.fp16 is False


def test_unknown_section_rejected():
    try:
        Config.from_dict({"nope": {"a": 1}})
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown section")


def test_unknown_key_rejected():
    try:
        Config.from_dict({"train": {"not_a_key": 1}})
    except ValueError as exc:
        assert "not_a_key" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown key")


def test_fp16_bf16_mutually_exclusive():
    cfg = Config()
    cfg.train.fp16 = True
    cfg.train.bf16 = True
    try:
        cfg.validate()
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_flat_dict_is_mlflow_friendly():
    flat = Config().to_flat_dict()
    assert flat["model.name"] == "openai/whisper-medium"
    assert all(isinstance(v, (str, int, float, bool)) or v is None for v in flat.values())


# --------------------------------------------------------------------- text


def test_normalize_collapses_elongation():
    # Sheng uses elongation expressively; we cap it so WER isn't noise.
    assert normalize_text("nomaaa") == "nomaa"


def test_normalize_lowercase_and_punctuation():
    out = normalize_text("Wakurugenzii, haha.")
    assert out == "wakurugenzii haha"


def test_normalize_keeps_apostrophes():
    assert "'" in normalize_text("it's a thing")


def test_normalize_handles_curly_apostrophe():
    assert normalize_text("it\u2019s") == normalize_text("it's")


def test_normalize_idempotent():
    s = normalize_text("Eeeh I was really going, you know!!")
    assert normalize_text(s) == s


def test_normalize_none():
    assert normalize_text(None) == ""


def test_tokens():
    assert tokens("mambo  ya  nguvu") == ["mambo", "ya", "nguvu"]


# ------------------------------------------------------------------ metrics


def test_perfect_prediction_scores_zero():
    refs = ["mambo ya nguvu", "i was going into the story"]
    rep = compute_metrics(refs, refs)
    assert rep.wer == 0.0
    assert rep.cer == 0.0


def test_wer_ignores_punctuation_and_case():
    rep = compute_metrics(["Mambo, ya nguvu."], ["mambo ya nguvu"])
    assert rep.wer == 0.0


def test_one_word_error():
    rep = compute_metrics(["mambo ya nguvu"], ["mambo ya sana"])
    assert abs(rep.wer - 1 / 3) < 1e-9


def test_mixed_wer_only_uses_codeswitched():
    refs = ["mambo ya nguvu sana", "i was going into the story"]
    hyps = ["mambo ya nguvu sana", "i was go into the story"]
    rep = compute_metrics(refs, hyps)
    # Overall WER is diluted by the perfect Swahili-only row.
    assert rep.wer > 0.0
    # mixed_wer considers only the row with English tokens.
    assert rep.mixed_wer > rep.wer
    assert rep.n_mixed_examples >= 1


def test_empty_hypothesis_does_not_crash():
    rep = compute_metrics(["mambo ya nguvu"], [""])
    assert rep.wer >= 1.0


def test_english_fraction_detects_switching():
    assert english_token_fraction("mambo ya nguvu") < 0.2
    assert english_token_fraction("i was going into the story") > 0.5


def test_length_mismatch_raises():
    try:
        compute_metrics(["a"], ["a", "b"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_report_serialises():
    rep = compute_metrics(["mambo"], ["mambo"])
    d = rep.to_dict()
    assert "per_example" in d and len(d["per_example"]) == 1
    m = rep.to_metrics_dict()
    assert all(isinstance(v, float) for v in m.values())


# --------------------------------------------------------------- augmentation


def test_speed_perturb_changes_length():
    import numpy as np

    from whisper_sheng.data import augment_speed
    import random

    arr = np.sin(np.linspace(0, 20, 16000)).astype(np.float32)
    out = augment_speed(arr, 16000, 0.9, random.Random(0))
    # factor < 1 means slower/longer
    assert len(out) > len(arr)
    assert out.dtype == np.float32


def test_speed_perturb_identity():
    import numpy as np
    import random

    from whisper_sheng.data import augment_speed

    arr = np.ones(100, dtype=np.float32)
    out = augment_speed(arr, 16000, 1.0, random.Random(0))
    assert len(out) == len(arr)


def test_spec_augment_preserves_shape():
    import numpy as np
    import random

    from whisper_sheng.data import spec_augment

    cfg = Config()
    # Use varied values: a constant array would make mean-filling a no-op
    # and hide a genuinely broken mask.
    feats = np.random.RandomState(0).randn(80, 3000).astype(np.float32)
    out = spec_augment(feats, cfg, random.Random(0))
    assert out.shape == feats.shape
    assert not np.array_equal(out, feats), "spec_augment did not modify features"
    # Only masked cells change, and they become the global mean.
    assert np.isclose(float(out.mean()), float(feats.mean()), atol=0.2)


def test_spec_augment_is_deterministic_for_seed():
    import numpy as np
    import random

    from whisper_sheng.data import spec_augment

    cfg = Config()
    feats = np.random.RandomState(1).randn(80, 500).astype(np.float32)
    a = spec_augment(feats, cfg, random.Random(7))
    b = spec_augment(feats, cfg, random.Random(7))
    assert np.array_equal(a, b)


def test_spec_augment_tolerates_narrow_input():
    import numpy as np
    import random

    from whisper_sheng.data import spec_augment

    cfg = Config()
    # Time-mask param larger than the clip must not crash or produce
    # out-of-range slicing.
    feats = np.random.RandomState(2).randn(80, 3).astype(np.float32)
    out = spec_augment(feats, cfg, random.Random(0))
    assert out.shape == feats.shape


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
