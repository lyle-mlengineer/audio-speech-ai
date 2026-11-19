from __future__ import annotations

import json

import pytest

from sheng_qwentts.paths import Paths
from sheng_qwentts.progress import BEGIN, DONE, END, FAILED, RUNNING, STAGES, Progress


@pytest.fixture()
def paths(tmp_path) -> Paths:
    return Paths(tmp_path / "artifacts", repo_progress=tmp_path / "progress.md").ensure()


def test_starts_empty(paths):
    prog = Progress(paths, "sheng_06b")
    assert all(prog.status(stage) == "todo" for stage, _ in STAGES)
    assert not prog.is_done("data")


def test_mark_persists_and_is_readable_again(paths):
    prog = Progress(paths, "sheng_06b")
    prog.mark("data", DONE, "38 train / 4 val")
    again = Progress(paths, "sheng_06b")
    assert again.is_done("data")
    assert again.get("data", "detail") == "38 train / 4 val"
    state = json.loads(paths.state_json.read_text(encoding="utf-8"))
    assert state["run"] == "sheng_06b"
    assert state["stages"]["data"]["status"] == DONE


def test_unknown_stage_is_rejected(paths):
    with pytest.raises(ValueError, match="unknown stage"):
        Progress(paths, "run").mark("tears", DONE)


def test_markdown_table_has_every_stage(paths):
    prog = Progress(paths, "sheng_06b")
    prog.mark("train", RUNNING, "step 40")
    prog.mark("data", DONE, "ok")
    prog.mark("eval", FAILED, "silent wavs")
    md = paths.artifacts_progress_md.read_text(encoding="utf-8")
    for stage, _ in STAGES:
        assert f"`{stage}`" in md
    assert "**done**" in md and "**running**" in md and "**FAILED**" in md


def test_console_view_is_compact(paths):
    prog = Progress(paths, "sheng_06b")
    text = prog.render_console()
    assert "sheng_06b" in text
    assert len(text.splitlines()) == len(STAGES) + 1


def test_repo_progress_md_keeps_the_hand_written_header(paths, monkeypatch):
    paths.repo_progress_md.write_text(
        "# My notes\n\nhuman text\n\n" + BEGIN + "\nstale\n" + END + "\n\nfooter\n",
        encoding="utf-8",
    )
    prog = Progress(paths, "sheng_06b")
    prog.mark("setup", DONE)
    text = paths.repo_progress_md.read_text(encoding="utf-8")
    assert text.startswith("# My notes")
    assert "stale" not in text
    assert "`setup`" in text
    assert text.rstrip().endswith("footer")


def test_corrupt_state_file_does_not_explode(paths):
    paths.state_json.write_text("{not json", encoding="utf-8")
    prog = Progress(paths, "sheng_06b")
    assert prog.status("data") == "todo"


def test_note_round_trip(paths):
    prog = Progress(paths, "sheng_06b")
    prog.note("train", {"steps": 240})
    assert Progress(paths, "sheng_06b").state["train"]["steps"] == 240
