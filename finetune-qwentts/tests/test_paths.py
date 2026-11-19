from __future__ import annotations

import json
from pathlib import Path

import pytest

from sheng_qwentts.paths import (
    Paths,
    all_checkpoints,
    checkpoint_step,
    is_checkpoint,
    latest_checkpoint,
    resolve_artifacts,
)


@pytest.fixture()
def paths(tmp_path, monkeypatch) -> Paths:
    monkeypatch.delenv("SHENG_QWENTTS_ARTIFACTS", raising=False)
    return Paths(tmp_path / "artifacts").ensure()


def _fake_checkpoint(path, step: int = 1):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"x")
    (path / "train_state.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    return path


def test_artifacts_from_flag_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("SHENG_QWENTTS_ARTIFACTS", str(tmp_path / "env"))
    assert resolve_artifacts(tmp_path / "flag") == (tmp_path / "flag").resolve()


def test_artifacts_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SHENG_QWENTTS_ARTIFACTS", str(tmp_path / "env"))
    assert resolve_artifacts(None) == (tmp_path / "env").resolve()


def test_artifacts_fallback_is_repo_artifacts(monkeypatch):
    monkeypatch.delenv("SHENG_QWENTTS_ARTIFACTS", raising=False)
    resolved = resolve_artifacts(None)
    assert resolved.name == "artifacts"


def test_layout_is_under_the_artifacts_root(paths):
    assert paths.wav_dir.parent == paths.data_dir
    assert paths.checkpoint_dir("run").parent == paths.checkpoint_root
    assert str(paths.train_codes_jsonl).endswith("train_with_codes.jsonl")


def test_is_checkpoint_needs_both_files(paths):
    step = paths.step_dir("run", 10)
    step.mkdir(parents=True)
    assert not is_checkpoint(step)
    (step / "model.safetensors").write_bytes(b"x")
    assert not is_checkpoint(step)
    (step / "train_state.json").write_text("{}", encoding="utf-8")
    assert is_checkpoint(step)


def test_no_checkpoint_yet(paths):
    assert latest_checkpoint(paths, "run") is None
    assert all_checkpoints(paths, "run") == []


def test_latest_pointer_wins(paths):
    first = _fake_checkpoint(paths.step_dir("run", 5))
    second = _fake_checkpoint(paths.step_dir("run", 9))
    paths.latest_pointer("run").write_text(second.name + "\n", encoding="utf-8")
    assert latest_checkpoint(paths, "run") == second
    assert all_checkpoints(paths, "run") == [first, second]


def test_latest_pointer_that_lies_falls_back_to_highest_step(paths):
    _fake_checkpoint(paths.step_dir("run", 5))
    best = _fake_checkpoint(paths.step_dir("run", 7))
    paths.latest_pointer("run").write_text("step_0000999\n", encoding="utf-8")
    assert latest_checkpoint(paths, "run") == best


def test_broken_newest_step_is_skipped(paths):
    good = _fake_checkpoint(paths.step_dir("run", 5))
    paths.step_dir("run", 6).mkdir(parents=True)  # died mid-save
    assert latest_checkpoint(paths, "run") == good


def test_checkpoint_step():
    assert checkpoint_step(Path("step_0000040")) == 40
    assert checkpoint_step(Path("step_0000040") / "nested") is None
    assert checkpoint_step(Path("latest")) is None
    assert checkpoint_step(Path("step_notanumber")) is None
