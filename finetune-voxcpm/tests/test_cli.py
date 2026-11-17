"""CLI surface: every subcommand is reachable and safe on an empty artifacts root."""

from __future__ import annotations

import json

import pytest

from sheng_voxcpm import cli


def _parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_subcommands_exist():
    assert set(cli.COMMANDS) == {"env", "data", "model", "train", "eval", "export", "status", "doctor", "all"}


def test_defaults():
    args = _parse("train")
    assert args.run == "lora_r32"
    assert args.steps == cli.DEFAULT_STEPS
    assert args.lora_r == cli.LORA_R
    assert args.lora_alpha is None
    assert args.profile is None  # decided from real VRAM
    assert args.overrides == []


def test_train_overrides_collected():
    args = _parse("train", "--set", "batch_size=1", "--set", "max_batch_tokens=4096")
    assert args.overrides == ["batch_size=1", "max_batch_tokens=4096"]
    opts = cli._train_options(args)
    assert opts.overrides == ["batch_size=1", "max_batch_tokens=4096"]


def test_eval_steps_are_optional():
    assert _parse("eval").steps is None
    assert _parse("eval", "--steps", "100", "300").steps == [100, 300]


def test_all_namespace_is_accepted_by_train_options():
    all_args = _parse("all", "--steps", "50", "--profile", "t4", "--allow-small-gpu")
    train_args = cli._train_args(all_args)
    opts = cli._train_options(train_args)
    assert opts.steps == 50 and opts.profile == "t4" and opts.allow_small_gpu
    assert opts.skip_prereqs is True  # env/data/model already ran inside `all`


def test_unknown_command_exits_with_usage():
    with pytest.raises(SystemExit):
        _parse("wat")


def test_status_json_on_empty_artifacts(tmp_path, capsys):
    code = cli.main(["status", "--artifacts", str(tmp_path / "artifacts"), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == {}
    assert payload["model_downloaded"] is False
    assert payload["voxcpm_repo"] is False
    assert payload["manifests"] == {"train": False, "val": False}
    assert payload["next"].startswith("`sheng-voxcpm env`")
    # progress.md is seeded into the artifacts root too
    assert (tmp_path / "artifacts" / "progress" / "progress.md").exists()


def test_status_markdown(tmp_path, capsys):
    code = cli.main(["status", "--artifacts", str(tmp_path / "artifacts")])
    assert code == 0
    out = capsys.readouterr().out
    assert "| stage  | status |" in out
    assert "| setup  | todo" in out
    assert "next:" in out


def test_doctor_runs_and_reports_checks(tmp_path, capsys):
    code = cli.main(["doctor", "--artifacts", str(tmp_path / "artifacts")])
    assert code in (0, 1)  # 1 in CI (no GPU), 0 on a working Colab GPU box
    out = capsys.readouterr().out
    assert "python 3.10-3.12" in out
    assert "CUDA GPU" in out
    assert "VoxCPM repo" in out


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
    from sheng_voxcpm.progress import Progress

    progress = Progress(paths)
    for stage in ("setup", "data", "model"):
        progress.event(stage, "done", note="x")
    args = _parse("all")

    assert cli._stage_already_done(paths, args, "data", progress) is True
    assert cli._stage_already_done(paths, args, "setup", progress) is False  # no marker yet

    paths.train_script.parent.mkdir(parents=True, exist_ok=True)
    paths.train_script.write_text("# stub", encoding="utf-8")
    marker = paths.progress_dir / "setup-marker.json"
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli.E, "marker_path", lambda: marker)
    monkeypatch.setattr(cli.E, "missing_imports", lambda: [])
    assert cli._setup_ready(paths) is True
    assert cli._stage_already_done(paths, args, "setup", progress) is True
    monkeypatch.setattr(cli.E, "missing_imports", lambda: ["torch"])
    assert cli._stage_already_done(paths, args, "setup", progress) is False


def test_train_stage_done_only_up_to_requested_steps(paths):
    from sheng_voxcpm.progress import Progress

    progress = Progress(paths)
    progress.event("train", "done", note="step 100")
    ckpt = paths.checkpoint_dir("lora_r32") / "latest"
    ckpt.mkdir(parents=True)
    (ckpt / "training_state.json").write_text('{"step": 100}', encoding="utf-8")

    assert cli._stage_already_done(paths, _parse("all", "--steps", "100"), "train", progress) is True
    assert cli._stage_already_done(paths, _parse("all", "--steps", "600"), "train", progress) is False
    assert cli._stage_already_done(paths, _parse("all", "--force"), "train", progress) is False
