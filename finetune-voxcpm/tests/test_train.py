"""Checkpoint discovery, resume and failure diagnosis (no torch needed)."""

from __future__ import annotations

import json

import pytest

from sheng_voxcpm import train as T
from sheng_voxcpm.config import DEFAULT_STEPS


def _ckpt(root, step: int, *, latest: bool = False) -> "object":
    d = root / (("latest" if latest else f"step_{step:07d}"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "training_state.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    (d / "lora_weights.safetensors").write_bytes(b"x")
    return d


def test_checkpoints_are_sorted_and_exclude_latest(tmp_path):
    _ckpt(tmp_path, 200)
    _ckpt(tmp_path, 100)
    _ckpt(tmp_path, 300, latest=True)
    (tmp_path / "train.log").write_text("noise", encoding="utf-8")

    found = T.checkpoints(tmp_path)
    assert [s for s, _ in found] == [100, 200]


def test_missing_dir_is_empty(tmp_path):
    assert T.checkpoints(tmp_path / "nope") == []
    assert T.latest_checkpoint(tmp_path / "nope") is None
    assert T.resume_step(tmp_path / "nope") == 0


def test_latest_checkpoint_prefers_latest_copy(tmp_path):
    _ckpt(tmp_path, 100)
    latest = _ckpt(tmp_path, 500, latest=True)
    assert T.latest_checkpoint(tmp_path) == latest
    assert T.resume_step(tmp_path) == 500


def test_latest_falls_back_to_highest_step(tmp_path):
    _ckpt(tmp_path, 100)
    _ckpt(tmp_path, 700)
    (tmp_path / "latest").mkdir()  # empty dir must not count
    assert T.latest_checkpoint(tmp_path) == tmp_path / "step_0000700"
    assert T.resume_step(tmp_path) == 700


def test_resume_step_survives_garbage_state(tmp_path):
    d = _ckpt(tmp_path, 100, latest=True)
    (d / "training_state.json").write_text("{not json", encoding="utf-8")
    assert T.resume_step(tmp_path) == 0


def test_archive_run_moves_the_directory(tmp_path):
    _ckpt(tmp_path / "lora_r32", 100)
    archived = T.archive_run(tmp_path / "lora_r32")
    assert archived is not None and archived.is_dir()
    assert archived.name.startswith("lora_r32_archive_")
    assert not (tmp_path / "lora_r32").exists()
    assert T.archive_run(tmp_path / "lora_r32") is None  # nothing to archive now


@pytest.mark.parametrize(
    "raw,expected",
    [("batch_size=1", ("batch_size", 1)), ("learning_rate=5e-5", ("learning_rate", 5e-05)), ('ref="x"', ("ref", "x"))],
)
def test_parse_override(raw, expected):
    assert T._parse_override(raw) == expected


def test_parse_override_rejects_garbage():
    with pytest.raises(ValueError):
        T._parse_override("batch_size")


def test_diagnose_maps_common_colab_failures(tmp_path):
    log = tmp_path / "console.log"
    log.write_text("RuntimeError: CUDA error: out of memory", encoding="utf-8")
    assert "OOM" in T._diagnose(log)

    log.write_text("AssertionError: sample_rate mismatch: config says 48000", encoding="utf-8")
    assert "sample_rate" in T._diagnose(log)

    log.write_text("ModuleNotFoundError: No module named 'tensorboardX'", encoding="utf-8")
    assert "sheng-voxcpm env" in T._diagnose(log)

    log.write_text("something else", encoding="utf-8")
    assert "exited non-zero" in T._diagnose(log)
    assert "exited non-zero" in T._diagnose(tmp_path / "missing.log")


def test_defaults_are_sane():
    opts = T.TrainOptions()
    assert opts.run == "lora_r32"
    assert opts.steps == DEFAULT_STEPS
    assert opts.save_interval > 0 and opts.save_interval <= DEFAULT_STEPS
    assert opts.fresh is False and opts.dry_run is False


def test_prereq_assertion_lists_every_missing_piece(paths):
    with pytest.raises(SystemExit) as exc:
        T._assert_prereqs(paths)
    message = str(exc.value)
    assert "train manifest" in message
    assert "base model" in message
    assert "VoxCPM repo" in message


def test_dry_run_renders_config_without_gpu_or_downloads(paths):
    """--dry-run must work on a CPU-only box: it renders the YAML and stops."""
    result = T.train(paths, T.TrainOptions(steps=42, dry_run=True, overrides=["max_batch_tokens=2048"]), log=lambda *_: None)
    assert result["ok"] is True and result["dry_run"] is True
    text = paths.run_config("lora_r32").read_text(encoding="utf-8")
    assert "num_iters: 42" in text
    assert "max_batch_tokens: 2048" in text
    assert "learning_rate: 0.0001" in text  # plain decimals, valid YAML 1.1 floats
    assert not paths.state_json.exists() or "running" not in paths.state_json.read_text(encoding="utf-8")
