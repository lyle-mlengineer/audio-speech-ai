"""Training config: profiles, invariants, rendered YAML, trainer argv."""

from __future__ import annotations

import json

import pytest

from sheng_chatterbox.config import (
    EFFECTIVE_BATCH,
    PROFILES,
    SMALL_GPU,
    TrainConfig,
    allowed_profiles,
    format_float,
    steps_for_hours,
)
from sheng_chatterbox.paths import DEFAULT_STEPS, Paths


def test_all_profiles_have_the_same_effective_batch():
    for name, profile in PROFILES.items():
        assert profile.batch_size * profile.grad_accum == EFFECTIVE_BATCH, name


def test_from_profile_validates_and_applies_overrides():
    cfg = TrainConfig.from_profile("l4", steps=42, learning_rate=1e-4)
    assert cfg.steps == 42 and cfg.learning_rate == 1e-4
    assert cfg.effective_batch == EFFECTIVE_BATCH
    assert cfg.dtype == "bf16"


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        TrainConfig.from_profile("h100")


def test_small_gpu_profiles_are_hidden_by_default():
    assert set(allowed_profiles()) == {"l4", "a100"}
    assert set(allowed_profiles(allow_small_gpu=True)) == set(PROFILES)
    assert SMALL_GPU == {"t4", "cpu"}


def test_t4_must_use_fp16():
    """Turing has no native bf16 - the trainer autocasts to bf16 if we let it."""
    assert PROFILES["t4"].dtype == "fp16"
    cfg = TrainConfig.from_profile("t4")
    cfg.dtype = "bf16"
    with pytest.raises(ValueError, match="T4"):
        cfg.validate()


def test_effective_batch_is_enforced():
    cfg = TrainConfig.from_profile("l4")
    cfg.batch_size = 2
    with pytest.raises(ValueError, match="effective batch"):
        cfg.validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", 0),
        ("learning_rate", 0.5),
        ("dtype", "int8"),
        ("model_variant", "klingon"),
        ("save_interval", 1),
        ("keep_last", 0),
        ("cond_seconds", 30.0),
    ],
)
def test_bad_values_are_rejected(field, value):
    cfg = TrainConfig.from_profile("l4")
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_format_float_avoids_scientific_notation():
    assert format_float(5e-05) == "0.00005"
    assert format_float(1e-4) == "0.0001"
    assert format_float(1.0) == "1.0"
    assert format_float(0.01) == "0.01"
    # and it round-trips
    assert float(format_float(5e-05)) == 5e-05


def test_steps_for_hours():
    assert steps_for_hours(0.5, 200) == 100
    assert steps_for_hours(0.0, 200) == 1


def test_trainer_args_cover_every_trainer_flag(tmp_path):
    paths = Paths(tmp_path / "artifacts").ensure()
    cfg = TrainConfig.from_profile("a100", steps=300)
    argv = cfg.trainer_args(paths)

    assert "--train-manifest" in argv and str(paths.train_manifest) in argv
    assert argv[argv.index("--pretrained-dir") + 1] == str(paths.pretrained_dir)
    assert argv[argv.index("--base-t3") + 1] == "t3_mtl23ls_v2.safetensors"
    assert argv[argv.index("--steps") + 1] == "300"
    assert argv[argv.index("--batch-size") + 1] == "8"
    assert argv[argv.index("--grad-accum") + 1] == "1"
    assert argv[argv.index("--lr") + 1] == "0.00005"
    # no val manifest on disk yet -> empty string, never a dangling path
    assert argv[argv.index("--val-manifest") + 1] == ""

    paths.val_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.val_manifest.write_text("", encoding="utf-8")
    assert cfg.trainer_args(paths)[argv.index("--val-manifest") + 1] == str(paths.val_manifest)


def test_render_yaml_is_parseable_and_complete(paths, mini_yaml):
    cfg = TrainConfig.from_profile("l4", steps=DEFAULT_STEPS)
    text = cfg.render_yaml(paths)
    assert "\t" not in text
    assert "nan" not in text.lower()
    json.dumps(text)  # plain text, no odd escapes

    doc = mini_yaml(text)
    assert doc["run"] == cfg.run
    assert doc["profile"] == "l4"
    assert doc["steps"] == DEFAULT_STEPS
    assert doc["batch_size"] * doc["grad_accum"] == EFFECTIVE_BATCH
    assert doc["dtype"] == "bf16"
    assert doc["model_variant"] == "mtl"
    assert doc["language_id"] == "sw"
    assert doc["pretrained_dir"].endswith("pretrained/chatterbox")
    assert doc["base_t3"] == "t3_mtl23ls_v2.safetensors"
    assert doc["save_dir"].endswith("checkpoints/sheng_t3")
    assert doc["learning_rate"] == 5e-05


def test_write_yaml_lands_in_the_run_config(paths):
    cfg = TrainConfig.from_profile("l4")
    written = cfg.write_yaml(paths)
    assert written == paths.config_path("sheng_t3")
    assert written.read_text(encoding="utf-8").startswith("# rendered by sheng-chatterbox")


@pytest.mark.parametrize("profile", ["l4", "a100", "t4"])
def test_template_matches_renderer(paths, mini_yaml, profile):
    """configs/chatterbox_sheng_<profile>.yaml documents exactly what config.py renders."""
    template_path = paths.artifacts.parent / "configs" / f"chatterbox_sheng_{profile}.yaml"
    if not template_path.exists():  # tests run from the repo root normally
        from sheng_chatterbox.paths import REPO_ROOT

        template_path = REPO_ROOT / "configs" / f"chatterbox_sheng_{profile}.yaml"
    assert template_path.exists(), f"missing template {template_path}"

    raw = template_path.read_text(encoding="utf-8").replace("${ARTIFACTS}", paths.artifacts.as_posix())
    template = mini_yaml(raw)
    rendered = mini_yaml(TrainConfig.from_profile(profile).render_yaml(paths))

    for key in (
        "run",
        "profile",
        "batch_size",
        "grad_accum",
        "steps",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "max_grad_norm",
        "log_interval",
        "valid_interval",
        "save_interval",
        "keep_last",
        "max_text_len",
        "max_speech_len",
        "cond_seconds",
        "emotion_adv",
        "model_variant",
        "language_id",
        "dtype",
        "seed",
        "tensorboard",
        "pretrained_dir",
        "base_t3",
        "train_manifest",
        "save_dir",
        "cache_dir",
    ):
        assert template[key] == rendered[key], f"{profile}: {key} {template[key]!r} != {rendered[key]!r}"

    # val_manifest is written as "" in the template (no manifests exist there yet).
    assert template["val_manifest"] == ""
    assert rendered["val_manifest"] in ("", paths.val_manifest.as_posix())
