"""Progress journal: survives a lost VM, never clobbers hand-written notes."""

from __future__ import annotations

import json

from sheng_voxcpm.progress import BEGIN, END, STAGES, Progress, seed_progress_md


def test_event_writes_three_stores(paths):
    p = Progress(paths)
    p.event("data", "done", note="38 train / 4 val clips")

    state = json.loads(paths.state_json.read_text(encoding="utf-8"))
    assert state["stages"]["data"]["status"] == "done"
    assert state["stages"]["data"]["note"] == "38 train / 4 val clips"
    assert state["updated"] and state["host"]["python"]

    events = [json.loads(line) for line in paths.runs_jsonl.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["stage"] == "data" and events[-1]["status"] == "done"

    md = paths.repo_progress_md.read_text(encoding="utf-8")
    assert BEGIN in md and END in md
    assert "| 2 | `data` |" in md


def test_unknown_stage_rejected(paths):
    try:
        Progress(paths).event("nope", "done")
    except KeyError as exc:
        assert "unknown stage" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_running_status_and_current_stage(paths):
    p = Progress(paths)
    for stage in ("setup", "data", "model"):
        p.event(stage, "done", note="x")
    p.event("train", "running", note="profile=l4")
    state = p.load()
    assert state["current_stage"] == "train"
    assert not p.is_done("train")
    assert "training was interrupted" in p.next_step()

    p.event("train", "done", note="step 600")
    assert p.is_done("train")
    assert p.load().get("current_stage") is None
    assert "sheng-voxcpm eval" in p.next_step()


def test_next_step_walks_the_stages(paths):
    p = Progress(paths)
    assert p.next_step().startswith("`sheng-voxcpm env`")
    for stage in STAGES:
        p.event(stage, "done", note="x")
    assert "All stages done" in p.next_step()


def test_hand_written_notes_survive_updates(paths):
    p = Progress(paths)
    paths.repo_progress_md.write_text("# My notes\n\n- keep me\n", encoding="utf-8")
    p.event("setup", "done", note="voxcpm abc123")

    text = paths.repo_progress_md.read_text(encoding="utf-8")
    assert text.startswith("# My notes")
    assert "- keep me" in text
    assert BEGIN in text and END in text

    p.event("data", "done", note="clips")
    text2 = paths.repo_progress_md.read_text(encoding="utf-8")
    assert text2.count(BEGIN) == 1 and text2.count(END) == 1
    assert "- keep me" in text2


def test_heartbeat_updates_step_without_event(paths):
    p = Progress(paths)
    p.heartbeat("lora_r32", 250, note="loss/diff=0.41", gpu={"name": "L4"})
    state = p.load()
    assert state["current_step"] == 250
    assert state["heartbeat_note"] == "loss/diff=0.41"
    assert not paths.runs_jsonl.exists()
    assert "Current step:** `250`" in paths.repo_progress_md.read_text(encoding="utf-8")


def test_checkpoint_block_rendered(paths):
    p = Progress(paths)
    p.set_state(checkpoint={"step": 300, "path": "/ckpt/step_0000300", "loss_diff": 0.4, "loss_stop": 0.01})
    md = p.render_markdown()
    assert "step_0000300" in md and "0.4" in md


def test_notes_are_capped(paths):
    p = Progress(paths)
    for i in range(60):
        p.add_note(f"note {i}")
    notes = p.load()["notes"]
    assert len(notes) == 40
    assert notes[-1].endswith("note 59")


def test_seed_creates_repo_progress_md(paths):
    assert not paths.repo_progress_md.exists()
    out = seed_progress_md(paths)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    for must in ("# Progress - VoxCPM2 x Sheng", "After a Colab disconnect", "BEGIN:AUTO", "Notes / decisions"):
        assert must in text


def test_tail_events(paths):
    p = Progress(paths)
    p.event("setup", "done", note="a")
    p.event("data", "done", note="b")
    tail = p.tail_events(1)
    assert len(tail) == 1 and tail[0]["note"] == "b"
