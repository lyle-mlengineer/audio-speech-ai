from __future__ import annotations

import json

import pytest

from sauti_sheng.paths import Paths
from sauti_sheng.progress import BEGIN, DONE, END, FAILED, RUNNING, STAGES, Progress


@pytest.fixture()
def paths(tmp_path) -> Paths:
    return Paths(tmp_path / "artifacts", repo_progress=tmp_path / "progress.md").ensure()


def test_starts_empty(paths):
    prog = Progress(paths, "sheng_abel")
    assert all(prog.status(stage) == "todo" for stage, _ in STAGES)
    assert not prog.is_done("data")


def test_mark_persists(paths):
    Progress(paths, "sheng_abel").mark("data", DONE, "38 train / 4 val")
    again = Progress(paths, "sheng_abel")
    assert again.is_done("data")
    assert again.get("data", "detail") == "38 train / 4 val"
    state = json.loads(paths.state_json.read_text(encoding="utf-8"))
    assert state["stages"]["data"]["status"] == DONE


def test_unknown_stage_is_rejected(paths):
    with pytest.raises(ValueError, match="unknown stage"):
        Progress(paths, "run").mark("tears", DONE)


def test_markdown_has_every_stage(paths):
    prog = Progress(paths, "sheng_abel")
    prog.mark("train", RUNNING, "epoch 1")
    md = prog.render_markdown()
    for stage, _ in STAGES:
        assert f"`{stage}`" in md
    assert "epoch 1" in md


def test_console_render_marks_status(paths):
    prog = Progress(paths, "sheng_abel")
    prog.mark("model", FAILED, "no space")
    text = prog.render_console()
    assert "[!] 3. model" in text
    assert "no space" in text


def test_mirrors_into_repo_between_markers(paths):
    paths.repo_progress_md.write_text(f"# notes\n\n{BEGIN}\nold\n{END}\n\ntail\n", encoding="utf-8")
    Progress(paths, "sheng_abel").mark("data", DONE, "42 clips")
    text = paths.repo_progress_md.read_text(encoding="utf-8")
    assert text.startswith("# notes")
    assert text.rstrip().endswith("tail")
    assert "42 clips" in text
    assert "old" not in text


def test_creates_repo_progress_when_missing(paths):
    Progress(paths, "sheng_abel").mark("data", DONE)
    assert paths.repo_progress_md.is_file()
    assert "| 2 | `data` |" in paths.repo_progress_md.read_text(encoding="utf-8")
