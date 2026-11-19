"""Progress journal: it exists so a lost VM is not a lost run."""

from __future__ import annotations

import json

from sheng_voiceclone import progress


def test_starts_empty_and_names_the_first_stage(art):
    assert progress.next_stage(art) == "setup"
    assert progress.status_of(art, "data") == "todo"


def test_record_advances_stage_and_next(art):
    progress.record(art, "setup", "done", detail="profile l4", run="r1")
    assert progress.status_of(art, "setup") == "done"
    assert progress.next_stage(art) == "data"
    state = json.loads(art.state_file.read_text(encoding="utf-8"))
    assert state["run"] == "r1" and state["stages"]["setup"]["detail"] == "profile l4"


def test_events_log_is_append_only(art):
    progress.record(art, "setup", "done")
    progress.record(art, "setup", "running")
    lines = art.events_file.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(x)["status"] for x in lines] == ["done", "running"]


def test_unknown_stage_is_rejected(art):
    try:
        progress.record(art, "nope", "done")
    except KeyError as exc:
        assert "unknown stage" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_failed_stage_is_not_skipped(art):
    progress.record(art, "setup", "failed", detail="no cuda")
    assert progress.next_stage(art) == "setup"


def test_summary_lists_every_stage_and_the_next_command(art):
    progress.record(art, "setup", "done", run="r1")
    text = progress.summary(art)
    for stage in progress.ORDER:
        assert stage in text
    assert "sheng-voiceclone data" in text


def test_all_done_summary_says_so(art):
    for stage in progress.ORDER:
        progress.record(art, stage, "done")
    assert progress.next_stage(art) is None
    assert "all stages done" in progress.summary(art)


def test_md_is_written_to_artifacts_and_hand_notes_survive(art):
    md = art.root / "progress.md"
    md.write_text("# Progress\n\nhand-written notes here\n\n", encoding="utf-8")
    progress.record(art, "data", "done", detail="38/4/4")
    text = md.read_text(encoding="utf-8")
    assert "hand-written notes here" in text
    assert progress.BEGIN in text and progress.END in text
    assert "| 2 | `data` |" in text and "38/4/4" in text
    # a second update replaces the block, not duplicates it
    progress.record(art, "model", "done")
    text2 = md.read_text(encoding="utf-8")
    assert text2.count(progress.BEGIN) == 1 and "model" in text2


def test_corrupt_state_falls_back_to_todo(art):
    art.state_file.write_text("{broken", encoding="utf-8")
    assert progress.status_of(art, "data") == "todo"


def test_repo_progress_md_is_not_touched_for_a_temp_root(art, monkeypatch):
    """Tests and demos use a temp artifacts root: they must not edit the repo file."""
    from sheng_voiceclone import paths as paths_mod

    repo_md = paths_mod.REPO_ROOT / "progress.md"
    before = repo_md.read_text(encoding="utf-8") if repo_md.is_file() else None
    progress.record(art, "data", "done", detail="temp root")
    after = repo_md.read_text(encoding="utf-8") if repo_md.is_file() else None
    assert before == after
    assert (art.root / "progress.md").is_file()


def test_repo_progress_md_is_mirrored_for_an_in_repo_root(tmp_path, monkeypatch):
    from sheng_voiceclone import paths as paths_mod

    inside = paths_mod.Paths(root=paths_mod.REPO_ROOT / "artifacts")
    monkeypatch.setitem(paths_mod.__dict__, "REPO_ROOT", paths_mod.REPO_ROOT)
    assert inside.root / "progress.md" in progress._mirror_targets(inside)
    assert paths_mod.REPO_ROOT / "progress.md" in progress._mirror_targets(inside)


def test_drive_root_is_mirrored(tmp_path):
    from sheng_voiceclone import paths as paths_mod

    drive = paths_mod.Paths(root=paths_mod.Path("/content/drive/MyDrive/vc"))
    assert paths_mod.REPO_ROOT / "progress.md" in progress._mirror_targets(drive)
