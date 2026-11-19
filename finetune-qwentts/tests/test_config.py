from __future__ import annotations

from pathlib import Path

import pytest

from sheng_qwentts.config import (
    EFFECTIVE_BATCH,
    PROFILES,
    TrainConfig,
    allowed_profiles,
    format_float,
)
from sheng_qwentts.paths import Paths


@pytest.fixture()
def paths(tmp_path) -> Paths:
    return Paths(tmp_path / "artifacts").ensure()


def test_profiles_keep_the_effective_batch():
    for profile in PROFILES.values():
        assert profile.batch_size * profile.grad_accum == EFFECTIVE_BATCH


@pytest.mark.parametrize("name", ["l4", "a100"])
def test_from_profile_uses_profile_defaults(name):
    cfg = TrainConfig.from_profile(name)
    p = PROFILES[name]
    assert (cfg.batch_size, cfg.grad_accum) == (p.batch_size, p.grad_accum)
    assert cfg.dtype == p.dtype
    assert cfg.effective_batch == EFFECTIVE_BATCH


def test_t4_must_use_float16():
    cfg = TrainConfig.from_profile("t4")
    assert cfg.dtype == "float16"
    with pytest.raises(ValueError, match="bf16"):
        TrainConfig.from_profile("t4", dtype="bfloat16")


def test_rejects_batch_that_breaks_the_invariant():
    with pytest.raises(ValueError, match="batch_size\\*grad_accum"):
        TrainConfig.from_profile("l4", batch_size=8)


@pytest.mark.parametrize("lr", [0.0, -1e-6, 5e-2])
def test_rejects_silly_learning_rates(lr):
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig.from_profile("l4", learning_rate=lr)


def test_rejects_float32_on_a_gpu_profile():
    with pytest.raises(ValueError, match="float32"):
        TrainConfig.from_profile("l4", dtype="float32")


def test_unknown_profile():
    with pytest.raises(ValueError, match="unknown profile"):
        TrainConfig.from_profile("h100")


def test_small_gpu_gate():
    assert "t4" not in allowed_profiles()
    assert "t4" in allowed_profiles(allow_small_gpu=True)


def test_trainer_args_point_at_the_artifacts(paths):
    cfg = TrainConfig.from_profile("l4", steps=12)
    argv = cfg.trainer_args(paths)
    joined = " ".join(argv)
    assert str(paths.model_dir) in joined
    assert str(paths.train_codes_jsonl) in joined
    assert str(paths.checkpoint_dir(cfg.run)) in joined
    assert "--steps" in argv and argv[argv.index("--steps") + 1] == "12"
    assert "--val-jsonl" not in argv  # the CLI appends it


def test_yaml_is_written_and_reloadable(paths):
    cfg = TrainConfig.from_profile("l4", steps=7)
    path = cfg.write_yaml(paths)
    text = path.read_text(encoding="utf-8")
    assert "profile: l4" in text
    assert "steps: 7" in text
    assert "learning_rate: 0.000002" in text
    assert str(paths.model_dir).replace("\\", "/") in text
    # no scientific notation, no tabs, no windows separators in the path block
    assert "e-0" not in text
    assert "\t" not in text
    assert "\\" not in text.split("resolved paths")[-1]


def test_format_float():
    assert format_float(2e-6) == "0.000002"
    assert format_float(0.00005) == "0.00005"
    assert format_float(1.0) == "1.0"
    assert format_float(0.01) == "0.01"


# --------------------------------------------------------------- templates
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _keys(text: str) -> set[str]:
    keys = set()
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" in line:
            keys.add(line.split(":", 1)[0].strip())
    return keys


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_reference_template_matches_what_is_rendered(profile, tmp_path):
    template = CONFIG_DIR / f"qwentts_sheng_{profile}.yaml"
    if not template.is_file():
        pytest.skip(f"no template for {profile}")
    rendered = TrainConfig.from_profile(profile).render_yaml(Paths(tmp_path / "a").ensure())
    assert _keys(template.read_text(encoding="utf-8")) == _keys(rendered)


@pytest.mark.parametrize("profile", ["l4", "t4", "a100"])
def test_templates_have_no_scientific_notation(profile):
    text = (CONFIG_DIR / f"qwentts_sheng_{profile}.yaml").read_text(encoding="utf-8")
    assert "e-0" not in text
