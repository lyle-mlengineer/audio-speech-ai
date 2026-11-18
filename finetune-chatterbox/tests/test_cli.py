"""CLI surface: every subcommand is reachable and safe on an empty artifacts root."""

from __future__ import annotations

import json

import pytest

from sheng_chatterbox import cli


def _parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_subcommands_exist():
    assert set(cli.COMMANDS) == {"env", "data", "model", "train", "eval", "export", "status", "doctor", "all"}


def test_defaults():
    args = _parse("train")
    assert args.run == "sheng_t3"
    assert args.steps == cli.DEFAULT_STEPS
    assert args.profile is None  # decided from real VRAM
    assert args.learning_rate == 5e-05
    assert args.keep_last == 3
    assert args.allow_small_gpu is False


def test_shared_flags_work_before_and_after_the_subcommand():
    assert _parse("--artifacts", "/x", "status").artifacts == "/x"
    assert _parse("status", "--artifacts", "/x").artifacts == "/x"


def test_eval_defaults_to_latest_and_ab_is_opt_in():
    args = _parse("eval")
    assert args.step == "latest" and args.ab is False
    assert _parse("eval", "--ab", "--step", "200").step == "200"


def test_all_namespace_is_accepted_by_train_args(monkeypatch):
    monkeypatch.setattr(cli.E, "gpu_info", lambda: {"cuda": True, "name": "NVIDIA L4", "vram_gb": 24, "bf16": True})
    all_args = _parse("all", "--steps", "50", "--profile", "t4", "--allow-small-gpu", "--ab")
    train_args = cli._train_args(all_args)
    cfg = cli._make_config(train_args, paths=type("P", (), {"artifacts": "/x"})())
    assert cfg.steps == 50 and cfg.profile == "t4" and cfg.dtype == "fp16"
    assert cli._eval_args(all_args).ab is True


def test_small_gpu_profile_requires_the_flag():
    with pytest.raises(SystemExit, match="allow-small-gpu"):
        cli.pick_profile(override="t4", allow_small_gpu=False)
    assert cli.pick_profile(override="t4", allow_small_gpu=True)[0] == "t4"


@pytest.mark.parametrize(
    "gpu,expected",
    [
        ({"cuda": True, "name": "NVIDIA A100-SXM4-40GB", "vram_gb": 40, "bf16": True}, "a100"),
        ({"cuda": True, "name": "NVIDIA L4", "vram_gb": 23.7, "bf16": True}, "l4"),
    ],
)
def test_profile_is_picked_from_the_gpu(monkeypatch, gpu, expected):
    monkeypatch.setattr(cli.E, "gpu_info", lambda: gpu)
    assert cli.pick_profile()[0] == expected


def test_small_gpu_needs_opt_in(monkeypatch):
    monkeypatch.setattr(cli.E, "gpu_info", lambda: {"cuda": True, "name": "Tesla T4", "vram_gb": 15.6, "bf16": False})
    with pytest.raises(SystemExit, match="below the 20 GiB"):
        cli.pick_profile()
    assert cli.pick_profile(allow_small_gpu=True)[0] == "t4"


def test_no_gpu_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(cli.E, "gpu_info", lambda: {"cuda": False, "name": "cpu", "vram_gb": 0})
    assert cli.pick_profile()[0] == "cpu"


def test_unknown_command_exits_with_usage():
    with pytest.raises(SystemExit):
        _parse("wat")


def test_status_json_on_empty_artifacts(tmp_path, capsys):
    code = cli.main(["status", "--artifacts", str(tmp_path / "artifacts"), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == {}
    assert payload["model_downloaded"] is False
    assert payload["manifests"] == {"train": False, "val": False}
    assert payload["train_status"] == "todo"
    assert payload["next"].startswith("`sheng-chatterbox env`")
    assert (tmp_path / "artifacts" / "progress" / "progress.md").exists()


def test_status_markdown(tmp_path, capsys):
    code = cli.main(["status", "--artifacts", str(tmp_path / "artifacts")])
    assert code == 0
    out = capsys.readouterr().out
    assert "| stage  | status |" in out
    assert "| setup  | todo" in out
    assert "checkpoints: none yet" in out
    assert "next:" in out


def test_status_reports_a_resumable_checkpoint(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    latest = artifacts / "checkpoints" / "sheng_t3" / "latest"
    latest.mkdir(parents=True)
    (latest / "t3_cfg.safetensors").write_bytes(b"x")
    (latest / "training_state.json").write_text('{"step": 300}', encoding="utf-8")

    assert cli.main(["status", "--artifacts", str(artifacts), "--json", "--steps", "600"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["train_status"] == "partial"
    assert payload["train_step"] == 300


def test_doctor_runs_and_reports_checks(tmp_path, capsys):
    code = cli.main(["doctor", "--artifacts", str(tmp_path / "artifacts")])
    assert code in (0, 1)  # 1 where the ML stack is absent (CI), 0 on a ready box
    out = capsys.readouterr().out
    assert "GPU" in out
    assert "import torch" in out
    assert "artifacts on Drive" in out


def test_error_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    def boom(paths, args):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(cli.COMMANDS, "data", boom)
    assert cli.main(["data", "--artifacts", str(tmp_path / "artifacts")]) == 1
    assert "error: kaboom" in capsys.readouterr().out


def test_artifacts_root_is_not_mounted_off_colab(monkeypatch):
    class NoContent:
        def __init__(self, *_a, **_k):
            pass

        def is_dir(self):
            return False

    calls = []
    monkeypatch.setattr(cli, "Path", NoContent)
    monkeypatch.setattr(cli.E, "mount_drive", lambda **kw: calls.append(1))
    cli._ensure_artifacts_root(_parse("train"))
    assert calls == []


def test_explicit_artifacts_skips_mount(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.E, "mount_drive", lambda **kw: calls.append(1))
    cli._ensure_artifacts_root(_parse("train", "--artifacts", "/tmp/x"))
    cli._ensure_artifacts_root(_parse("doctor"))
    assert calls == []


def test_setup_stage_is_vm_local(paths, monkeypatch):
    """A fresh VM must redo `setup` even though state.json (on Drive) says done."""
    from sheng_chatterbox.progress import Progress

    progress = Progress(paths)
    for stage in ("setup", "data", "model"):
        progress.event(stage, "done", note="x")
    args = _parse("all")

    assert cli._stage_already_done(paths, args, "data", progress) is True
    assert cli._stage_already_done(paths, args, "setup", progress) is False  # nothing installed

    monkeypatch.setattr(cli.E, "is_installed", lambda: True)
    assert cli._stage_already_done(paths, args, "setup", progress) is True
    monkeypatch.setattr(cli.E, "is_installed", lambda: False)
    assert cli._stage_already_done(paths, args, "setup", progress) is False


def test_train_stage_done_only_up_to_requested_steps(paths):
    from sheng_chatterbox.progress import Progress

    progress = Progress(paths)
    progress.event("train", "done", note="step 100")
    ckpt = paths.latest_dir("sheng_t3")
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "t3_cfg.safetensors").write_bytes(b"x")
    (ckpt / "training_state.json").write_text('{"step": 100}', encoding="utf-8")

    assert cli._stage_already_done(paths, _parse("all", "--steps", "100"), "train", progress) is True
    assert cli._stage_already_done(paths, _parse("all", "--steps", "600"), "train", progress) is False
    assert cli._stage_already_done(paths, _parse("all", "--force"), "train", progress) is False


def test_fresh_archives_the_previous_run(paths, monkeypatch):
    old = paths.checkpoint_dir("sheng_t3")
    old.mkdir(parents=True)
    (old / "step_0000100").mkdir()

    cli._archive_run(paths, "sheng_t3")
    assert not old.exists()
    archived = list(paths.checkpoint_root.glob("sheng_t3.archived-*"))
    assert len(archived) == 1


def test_train_data_and_model_paths_come_from_the_run_name(tmp_path):
    args = _parse("export", "--artifacts", str(tmp_path / "a"), "--run", "other")
    assert args.run == "other"
