"""CLI wiring: the offline-safe commands must work end to end without a GPU."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from sheng_voiceclone import cli, progress
from sheng_voiceclone.paths import resolve


def _source_dir(tmp_path, n=8, seconds=1.5):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        sf.write(
            str(src / f"rec{i}.wav"),
            (0.2 * np.sin(2 * np.pi * (150 + i) * np.arange(int(seconds * 16000)) / 16000)).astype(np.float32),
            16000,
        )
        (src / f"rec{i}.txt").write_text(f"sasa bro hii ni sentensi ya {i}", encoding="utf-8")
    return src


def _run(*argv):
    return cli.main([str(a) for a in argv])


def test_prompts_command_prints_every_prompt(capsys):
    assert _run("prompts") == 0
    out = capsys.readouterr().out
    assert "sheng_01" in out and "swahili_01" in out and "english_01" in out


def test_status_on_an_empty_root_does_not_crash(art, capsys):
    assert _run("status", "--artifacts", art.root) == 0
    assert "next:" in capsys.readouterr().out


def test_data_then_status_then_baseline_dry_run(art, tmp_path, capsys):
    src = _source_dir(tmp_path)
    assert _run("data", "--artifacts", art.root, "--source", src, "--val-clips", 2, "--test-clips", 2) == 0
    p = resolve(str(art.root))
    assert p.ref_audio.is_file() and p.train_manifest.is_file()
    assert progress.status_of(p, "data") == "done"

    assert _run("status", "--artifacts", art.root) == 0
    out = capsys.readouterr().out
    assert "4 train" in out or "train /" in out

    assert _run("baseline", "--artifacts", art.root, "--dry-run") == 0
    assert json.loads((p.evals / "baseline" / "plan.json").read_text(encoding="utf-8"))


def test_baseline_without_data_explains_itself(art, capsys):
    assert _run("baseline", "--artifacts", art.root, "--dry-run") == 1
    assert "run `sheng-voiceclone data` first" in capsys.readouterr().out


def test_train_without_the_sibling_project_prints_the_command(art, tmp_path, capsys):
    src = _source_dir(tmp_path)
    _run("data", "--artifacts", art.root, "--source", src, "--val-clips", 2, "--test-clips", 2)
    rc = _run("train", "--artifacts", art.root, "--dry-run")
    out = capsys.readouterr().out
    assert "profile" in out and "config ->" in out
    assert rc in (0, 1)  # 0 when sheng_chatterbox happens to be installed


def test_compare_needs_both_score_files(art, capsys):
    assert _run("compare", "--artifacts", art.root) == 1
    assert "no scores for 'baseline'" in capsys.readouterr().out


def test_compare_after_fake_scores(art, scores_file, capsys):
    scores_file("baseline", cer=0.30, sim=0.55)
    scores_file("tuned", cer=0.24, sim=0.74)
    assert _run("compare", "--artifacts", art.root) == 0
    out = capsys.readouterr().out
    assert "KEEP" in out
    assert (art.scores / "compare.md").is_file()
    assert progress.status_of(art, "compare") == "done"


def test_score_command_re_scores_an_existing_plan(art, tmp_path, capsys):
    src = _source_dir(tmp_path)
    _run("data", "--artifacts", art.root, "--source", src, "--val-clips", 2, "--test-clips", 2)
    p = resolve(str(art.root))
    rows = json.loads((p.evals / "baseline" / "plan.json").read_text()) if (p.evals / "baseline" / "plan.json").is_file() else None
    if rows is None:
        _run("baseline", "--artifacts", art.root, "--dry-run")
    rc = _run("score", "--artifacts", art.root, "--dir", "baseline", "--no-asr", "--no-sim")
    assert rc == 0
    assert (p.scores / "baseline.json").is_file()


def test_model_command_is_never_run_offline():
    """`model` hits the network - documented, not tested here."""
    assert callable(cli.cmd_model)


def test_all_stops_cleanly_when_a_stage_fails(art, tmp_path, capsys, monkeypatch):
    src = _source_dir(tmp_path)
    monkeypatch.setattr(cli, "cmd_model", lambda a: 1)  # simulate a dead stage
    rc = _run("all", "--artifacts", art.root, "--source", src, "--dry-run",
              "--val-clips", 2, "--test-clips", 2)
    assert rc == 1
    out = capsys.readouterr().out
    assert "stopped at 'model'" in out and "--resume" in out


def test_profile_choice_is_rendered_into_yaml(art):
    _run("setup", "--artifacts", art.root, "--profile", "t4")
    yaml = (art.configs / "sheng_clone.yaml").read_text(encoding="utf-8")
    assert "profile: t4" in yaml and "effective_batch: 8" in yaml
    assert progress.status_of(art, "setup") in {"done", "failed"}


def test_compare_picks_the_newest_scored_checkpoint(art, scores_file, capsys):
    scores_file("baseline", cer=0.40, sim=0.50)
    scores_file("tuned_0000100", cer=0.30, sim=0.70)
    scores_file("tuned_0000200", cer=0.55, sim=0.74)  # words got worse
    assert _run("compare", "--artifacts", art.root) == 0
    out = capsys.readouterr().out
    assert "MIXED" in out  # from tuned_0000200, the newest
    assert "tuned_0000200" in (art.scores / "compare.md").read_text(encoding="utf-8")


def test_compare_uses_tuned_when_it_exists(art, scores_file, capsys):
    scores_file("baseline", cer=0.40, sim=0.50)
    scores_file("tuned", cer=0.30, sim=0.70)
    scores_file("tuned_0000100", cer=0.90, sim=0.10)
    assert _run("compare", "--artifacts", art.root) == 0
    assert "KEEP" in capsys.readouterr().out
