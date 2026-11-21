"""Profiles and their rendered YAML: the numbers a run is reproducible from."""

from __future__ import annotations

import pytest

from sheng_voiceclone import config as cfg


def test_every_profile_has_the_same_effective_batch():
    for name in cfg.PROFILES:
        assert cfg.profile(name).effective_batch == cfg.EFFECTIVE_BATCH


def test_profile_lookup_and_errors():
    assert cfg.profile("l4").dtype == "bfloat16"
    with pytest.raises(KeyError, match="unknown profile"):
        cfg.profile("h100")


@pytest.mark.parametrize("vram,expected", [(8.0, "t4"), (16.0, "t4"), (24.0, "l4"), (40.0, "a100"), (80.0, "a100")])
def test_pick_profile_is_conservative(vram, expected):
    assert cfg.pick_profile(vram) == expected


def test_t4_is_the_only_profile_allowed_a_small_gpu():
    assert cfg.profile("t4").allow_small_gpu is True
    assert all(not cfg.profile(n).allow_small_gpu for n in ("l4", "a100"))


def test_override_applies_flags_and_keeps_the_rest():
    p = cfg.override(cfg.profile("l4"), steps=120, lr=1e-6, lora=True)
    assert (p.steps, p.lr, p.lora) == (120, 1e-6, True)
    assert p.batch_size == cfg.profile("l4").batch_size


def test_override_ignores_none_and_rejects_unknown():
    p = cfg.profile("l4")
    assert cfg.override(p, steps=None, lora=None) is p
    with pytest.raises(KeyError, match="unknown profile field"):
        cfg.override(p, nope=1)


def test_render_resolves_paths_into_the_run(art):
    text = cfg.render(cfg.profile("l4"), art, run="myrun")
    assert f"run: myrun" in text
    assert str(art.train_manifest) in text
    assert "effective_batch: 8" in text


def test_rendered_yaml_parses_back(art):
    p = cfg.write_yaml(cfg.profile("t4"), art, run="r1")
    parsed = cfg.parse_yaml(p.read_text(encoding="utf-8"))
    assert parsed["profile"] == "t4"
    assert parsed["steps"] == 200
    assert parsed["lr"] == pytest.approx(5e-5)
    assert parsed["lora"] is False
    assert parsed["train_manifest"].endswith("train.jsonl")


def test_parse_yaml_handles_bools_ints_floats_and_strings():
    parsed = cfg.parse_yaml("a: true\nb: 12\nc: 0.5\nd: hello\n# comment\ne: 3  # trailing\n")
    assert parsed == {"a": True, "b": 12, "c": 0.5, "d": "hello", "e": 3}


def test_repo_template_matches_the_l4_profile():
    """The committed reference template must not drift from the code."""
    from pathlib import Path

    tpl = Path(__file__).resolve().parents[1] / "configs" / "voice_clone_sheng_l4.yaml"
    parsed = cfg.parse_yaml(tpl.read_text(encoding="utf-8"))
    prof = cfg.profile("l4")
    for key in ("batch_size", "grad_accum", "steps", "lr", "weight_decay", "save_interval", "keep_last"):
        assert parsed[key] == getattr(prof, key), key
