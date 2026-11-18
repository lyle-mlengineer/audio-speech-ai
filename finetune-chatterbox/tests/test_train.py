"""Driver helpers: prerequisites, stop guard, log tailing, status bookkeeping."""

from __future__ import annotations

import json

import pytest

from sheng_chatterbox import train as T
from sheng_chatterbox.config import TrainConfig


def test_trainer_command_uses_rendered_config_and_resume(paths):
    cfg = TrainConfig.from_profile("cpu", steps=7, learning_rate=5e-05)
    argv = T._trainer_command(paths, cfg, resume=True, dry_run=True)
    assert argv[1:3] == ["-m", "sheng_chatterbox.train_t3"]
    assert argv[argv.index("--resume") + 1] == "auto"
    assert "--dry-run" in argv
    assert argv[argv.index("--steps") + 1] == "7"

    argv = T._trainer_command(paths, cfg, resume=False, dry_run=False)
    assert argv[argv.index("--resume") + 1] == "none"
    assert "--dry-run" not in argv


def test_variant_file_names():
    assert T.cfg_variant_file(TrainConfig.from_profile("l4")) == "t3_mtl23ls_v2.safetensors"
    assert T.cfg_variant_file(TrainConfig.from_profile("l4", model_variant="english")) == "t3_cfg.safetensors"


def test_prerequisites_list_every_missing_thing(paths):
    cfg = TrainConfig.from_profile("cpu")
    with pytest.raises(T.MissingPrerequisites) as exc:
        T.check_prerequisites(paths, cfg)
    message = str(exc.value)
    assert "train manifest" in message
    assert "sheng-chatterbox data" in message
    assert "sheng-chatterbox model" in message


def test_prerequisites_pass_when_everything_is_there(paths, monkeypatch):
    cfg = TrainConfig.from_profile("cpu")
    paths.train_manifest.write_text('{"audio": "a.wav", "text": "x"}\n', encoding="utf-8")
    for name in ("ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "t3_mtl23ls_v2.safetensors"):
        (paths.pretrained_dir / name).write_bytes(b"x")

    import types

    for name in ("torch", "chatterbox"):
        monkeypatch.setitem(__import__("sys").modules, name, types.ModuleType(name))
    T.check_prerequisites(paths, cfg)  # must not raise


def test_tail_returns_only_complete_lines(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("a\nb\npartial", encoding="utf-8")
    lines, offset = T.tail(log)
    assert lines == ["a", "b"]
    assert offset == len(b"a\nb\n")

    with log.open("a", encoding="utf-8") as f:
        f.write(" line\n")
    more, offset = T.tail(log, offset)
    assert more == ["partial line"]


def test_tail_missing_file_is_harmless(tmp_path):
    assert T.tail(tmp_path / "nope.log") == ([], 0)


def test_train_status_tracks_requested_steps(paths):
    assert T.train_status(paths, "r", 100) == ("todo", None)

    latest = paths.latest_dir("r")
    latest.mkdir(parents=True)
    (latest / "t3_cfg.safetensors").write_bytes(b"x")
    (latest / "training_state.json").write_text(json.dumps({"step": 100}), encoding="utf-8")

    assert T.train_status(paths, "r", 100) == ("done", 100)
    assert T.train_status(paths, "r", 600) == ("partial", 100)


def test_train_status_survives_a_broken_state_file(paths):
    latest = paths.latest_dir("r")
    latest.mkdir(parents=True)
    (latest / "t3_cfg.safetensors").write_bytes(b"x")
    (latest / "training_state.json").write_text("not json", encoding="utf-8")
    assert T.train_status(paths, "r", 100) == ("partial", 0)


def test_final_summary_classifies_outcomes(paths, monkeypatch):
    cfg = TrainConfig.from_profile("cpu", steps=100)
    latest = paths.latest_dir(cfg.run)
    latest.mkdir(parents=True)
    (latest / "t3_cfg.safetensors").write_bytes(b"x")
    (latest / "training_state.json").write_text(
        json.dumps({"step": 100, "metrics": {"loss": 2.5}}), encoding="utf-8"
    )

    done = T._final_summary(paths, cfg, 0, 100, None, None, 0.0, False)
    assert done["status"] == "done" and done["loss"] == 2.5 and done["step"] == 100

    partial_cfg = TrainConfig.from_profile("cpu", steps=600)
    partial = T._final_summary(paths, partial_cfg, 130, 100, None, None, 0.0, False)
    assert partial["status"] == "partial"

    failed_cfg = TrainConfig.from_profile("cpu", steps=600)
    (latest / "training_state.json").write_text(json.dumps({"step": 0}), encoding="utf-8")
    failed = T._final_summary(paths, failed_cfg, 1, 0, None, None, 0.0, False)
    assert failed["status"] == "failed"
    assert "exited with code 1" in failed["error"]


def test_dry_run_summary_is_not_done(paths):
    cfg = TrainConfig.from_profile("cpu", steps=10)
    summary = T._final_summary(paths, cfg, 0, 0, 0.0, None, 0.0, True)
    assert summary["status"] == "dry-run"


def test_send_stop_prefers_sigint(paths, monkeypatch):
    sent = []

    class FakeProc:
        def send_signal(self, sig):
            sent.append(sig)

        def terminate(self):
            sent.append("terminate")

    monkeypatch.setattr(T.os, "name", "posix")
    T._send_stop(FakeProc())
    assert sent and sent[0] != "terminate"


def test_tensorboard_flag_only_when_enabled(paths):
    on = TrainConfig.from_profile("l4", tensorboard=True)
    off = TrainConfig.from_profile("l4", tensorboard=False)
    assert "--tensorboard" in on.trainer_args(paths)
    assert "--tensorboard" not in off.trainer_args(paths)
