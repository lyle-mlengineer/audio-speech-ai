"""CLI surface: every subcommand is reachable and safe on an empty artifacts root."""

from __future__ import annotations

import json

import pytest

from kenya_voices import cli
from kenya_voices.paths import DEFAULT_RUN


def _parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_subcommands_exist():
    assert set(cli.COMMANDS) == {
        "env",
        "data",
        "model",
        "train",
        "eval",
        "export",
        "design",
        "status",
        "doctor",
        "datasets",
        "all",
    }


def test_train_defaults():
    args = _parse("train")
    assert args.run == DEFAULT_RUN
    assert args.steps == cli.DEFAULT_STEPS
    assert args.lora_r == cli.LORA_R
    assert args.lora_alpha is None  # -> same as r
    assert args.profile is None  # decided from real VRAM
    assert args.overrides == []


def test_data_defaults_match_the_preset():
    args = _parse("data")
    cfg = cli._data_config(args)
    assert cfg.resolved_dataset == "badrex/swahili-speech-400hr"
    assert cfg.max_speakers == 120
    assert 0 < cfg.caption_dropout < 1
    assert cfg.val_speaker_fraction > 0


def test_data_flags_reach_the_config():
    args = _parse(
        "data",
        "--speakers",
        "40",
        "--clips-per-speaker",
        "12",
        "--hours",
        "3.5",
        "--caption-dropout",
        "0.25",
        "--caption-variants",
        "2",
        "--preset",
        "kenyan-voices",
    )
    cfg = cli._data_config(args)
    assert cfg.max_speakers == 40
    assert cfg.max_clips_per_speaker == 12
    assert cfg.max_audio_hours == 3.5
    assert cfg.caption_dropout == 0.25
    assert cfg.caption_variants == 2
    assert cfg.resolved_dataset == "lyle-mlengineer/kenyan-voices"


def test_train_options_carry_the_data_config():
    args = _parse("train", "--steps", "50")
    opts = cli._train_options(args)
    assert opts.steps == 50
    assert opts.data is None  # `train` alone does not rebuild data


def test_all_namespace_is_accepted_by_train_options():
    args = _parse("all", "--steps", "50", "--speakers", "80", "--hours", "6")
    train_args = cli._train_args(args)
    opts = cli._train_options(train_args)
    assert opts.steps == 50
    assert opts.data.max_speakers == 80
    assert opts.data.max_audio_hours == 6.0


def test_unknown_command_exits_with_usage():
    with pytest.raises(SystemExit):
        _parse("nope")


def test_design_requires_caption_and_text():
    with pytest.raises(SystemExit):
        _parse("design", "--text", "Habari")
    args = _parse("design", "--caption", "A Kenyan male voice", "--text", "Habari yako")
    assert args.caption.startswith("A Kenyan")


def test_status_json_on_empty_artifacts(tmp_path, capsys):
    code = cli.main(["--artifacts", str(tmp_path / "a"), "status", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_downloaded"] is False
    assert payload["stages"] == {}  # nothing has run yet
    assert payload["next"].startswith("`")


def test_status_markdown(tmp_path, capsys):
    assert cli.main(["--artifacts", str(tmp_path / "a"), "status"]) == 0
    out = capsys.readouterr().out
    assert "artifacts :" in out
    assert "next:" in out


def test_doctor_runs_and_reports_checks(tmp_path, capsys):
    code = cli.main(["--artifacts", str(tmp_path / "a"), "doctor"])
    out = capsys.readouterr().out
    assert "CUDA GPU" in out
    assert "base model" in out
    assert code in (0, 1)  # 1 on a machine with no GPU/model - the point is it reports, not crashes


def test_datasets_lists_presets(tmp_path, capsys):
    assert cli.main(["--artifacts", str(tmp_path / "a"), "datasets"]) == 0
    out = capsys.readouterr().out
    assert "badrex" in out and "kenyan-voices" in out
    assert "(default)" in out


def test_train_dry_run_writes_a_config(tmp_path, capsys):
    code = cli.main(["--artifacts", str(tmp_path / "a"), "train", "--dry-run", "--steps", "10"])
    out = capsys.readouterr().out
    assert code == 0
    assert "dry-run" in out
    config = tmp_path / "a" / "configs" / f"{DEFAULT_RUN}.yaml"
    assert config.exists()
    text = config.read_text(encoding="utf-8")
    assert "sample_rate: 16000" in text
    assert "num_iters: 10" in text


def test_error_is_reported_not_raised(tmp_path, capsys):
    code = cli.main(["--artifacts", str(tmp_path / "a"), "eval", "--steps", "5"])
    out = capsys.readouterr().out
    assert code == 1
    assert "error:" in out


def test_artifacts_root_is_not_mounted_off_colab(monkeypatch):
    called = {"mount": 0}
    monkeypatch.setattr(cli.E, "mount_drive", lambda log=print: called.__setitem__("mount", called["mount"] + 1))
    monkeypatch.setattr(cli.Path, "is_dir", lambda self: False)
    cli._ensure_artifacts_root(cli.build_parser().parse_args(["status"]))
    assert called["mount"] == 0


def test_explicit_artifacts_skips_mount(tmp_path, monkeypatch):
    called = {"mount": 0}
    monkeypatch.setattr(cli.E, "mount_drive", lambda log=print: called.__setitem__("mount", called["mount"] + 1))
    cli._ensure_artifacts_root(cli.build_parser().parse_args(["--artifacts", str(tmp_path), "status"]))
    assert called["mount"] == 0


def test_setup_stage_is_vm_local(tmp_path, monkeypatch):
    from kenya_voices.paths import Paths

    paths = Paths(artifacts=tmp_path / "a", code_root=tmp_path)
    paths.ensure()
    monkeypatch.setattr(cli.E, "marker_path", lambda: tmp_path / "missing.json")
    assert cli._setup_ready(paths) is False


def test_train_stage_done_only_up_to_requested_steps(tmp_path):
    from kenya_voices.paths import Paths
    from kenya_voices.progress import Progress

    paths = Paths(artifacts=tmp_path / "a", code_root=tmp_path)
    paths.ensure()
    progress = Progress(paths)
    progress.event("train", "done", note="x")
    args = cli.build_parser().parse_args(["all", "--steps", "100"])
    assert cli._stage_already_done(paths, args, "train", progress) in (True, False)
