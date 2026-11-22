from __future__ import annotations

import pytest

from sauti_sheng.config import (
    DEFAULT_PROFILE,
    PROFILES,
    auto_profile,
    render_config,
    resolve_profile,
    updates_for,
    updates_per_epoch,
    total_updates,
    write_config,
)
from sauti_sheng.paths import Paths


def test_every_profile_has_effective_batch_four():
    for profile in PROFILES.values():
        assert profile.effective_batch == 4


def test_default_profile_is_t4():
    assert DEFAULT_PROFILE == "t4"
    assert PROFILES[DEFAULT_PROFILE].mixed_precision == "fp16"  # T4 has no bf16
    assert PROFILES[DEFAULT_PROFILE].effective_batch == 4


def test_auto_profile_maps_gpus():
    assert auto_profile("NVIDIA A100-SXM4-40GB", 40) == "a100"
    assert auto_profile("NVIDIA L4", 24) == "l4"
    assert auto_profile("Tesla T4", 16) == DEFAULT_PROFILE
    assert auto_profile("Tesla T4", 15) == DEFAULT_PROFILE
    assert auto_profile("Quadro M2000", 4) is None


def test_explicit_profile_wins():
    # Auto-detection would pick a100 for this GPU; the flag must override it.
    assert resolve_profile("l4", "NVIDIA A100-SXM4-40GB", 40).name == "l4"


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        resolve_profile("h100", "A100", 40)


def test_t4_runs_without_opt_in():
    # The T4 profile is the default runtime, so it must not need a flag.
    assert resolve_profile(None, "Tesla T4", 16).name == "t4"
    assert resolve_profile("t4", "Tesla T4", 16).name == "t4"
    # The legacy flag stays accepted (deprecated, no effect).
    assert resolve_profile("t4", "Tesla T4", 16, allow_small_gpu=True).name == "t4"


def test_vram_floor_is_enforced():
    with pytest.raises(RuntimeError, match="VRAM"):
        resolve_profile("a100", "NVIDIA A100", 20)


def test_no_gpu_raises():
    with pytest.raises(RuntimeError, match="no profile"):
        resolve_profile(None, None, None)


def test_update_maths():
    l4 = PROFILES["l4"]
    # 42 clips, batch 2 -> 21 batches, accum 2 -> 11 updates per epoch
    assert updates_for(42, 2, 2) == 11
    assert updates_per_epoch(42, l4) == 11
    assert total_updates(42, l4, epochs=10) == 110
    assert updates_for(0, 2, 2) == 0


def test_render_config_contains_paths_and_flags(tmp_path):
    paths = Paths(tmp_path / "artifacts").ensure()
    text = render_config(PROFILES[DEFAULT_PROFILE], "sheng_abel", paths, epochs=7)
    assert f"profile: {DEFAULT_PROFILE}" in text
    assert "epochs: 7" in text
    assert "effective_batch: 4" in text
    assert "mixed_precision: fp16" in text
    assert str(paths.sauti_ema_ckpt) in text
    assert str(paths.checkpoint_dir("sheng_abel")) in text


def test_write_config_lands_in_artifacts(tmp_path):
    paths = Paths(tmp_path / "artifacts").ensure()
    path = write_config(paths, PROFILES["a100"], "run1")
    assert path == paths.config_path("run1")
    assert path.is_file()
    assert "profile: a100" in path.read_text(encoding="utf-8")
