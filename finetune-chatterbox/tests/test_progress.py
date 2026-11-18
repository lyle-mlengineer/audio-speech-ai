"""Progress journal: survives a lost VM, never clobbers hand-written notes."""

from __future__ import annotations

import json

from sheng_chatterbox.progress import BEGIN, END, STAGES, Progress, seed_progress_md


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
    assert "sheng-chatterbox" in md  # commands are the real ones


def test_unknown_stage_rejected(paths):
    try:
        Progress(paths).event("nope", "done")
    except KeyError as exc:
        assert "unknown stage" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_stage_names_are_the_cli_stages():
    assert list(STAGES) == ["setup", "data", "model", "train", "eval", "export"]


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
    assert "sheng-chatterbox eval" in p.next_step()


def test_next_step_walks_the_stages(paths):
    p = Progress(paths)
    assert p.next_step().startswith("`sheng-chatterbox env`")
    for stage in STAGES:
        p.event(stage, "done", note="x")
    assert "All stages done" in p.next_step()


def test_partial_training_is_resumed_not_restarted(paths):
    p = Progress(paths)
    p.event("train", "partial", note="stopped at step 300/600")
    assert "--resume" in p.next_step()


def test_hand_written_notes_survive_updates(paths):
    p = Progress(paths)
    paths.repo_progress_md.write_text("# My notes\n\n- keep me\n", encoding="utf-8")
    p.event("setup", "done", note="chatterbox-tts 0.1.7")

    text = paths.repo_progress_md.read_text(encoding="utf-8")
    assert "# My notes" in text and "- keep me" in text
    assert BEGIN in text and END in text
    # re-rendering must not duplicate the block
    p.event("data", "done", note="x")
    text = paths.repo_progress_md.read_text(encoding="utf-8")
    assert text.count(BEGIN) == 1


def test_heartbeat_is_cheap_and_visible(paths):
    p = Progress(paths)
    p.heartbeat("sheng_t3", 250, note="step=0000250 loss=3.12")
    md = paths.repo_progress_md.read_text(encoding="utf-8")
    assert "`250`" in md
    assert "step=0000250 loss=3.12" in md
    assert not paths.runs_jsonl.exists() or "heartbeat" not in paths.runs_jsonl.read_text(encoding="utf-8")


def test_checkpoint_block_renders_losses(paths):
    p = Progress(paths)
    p.set_state(checkpoint={"step": 100, "path": "/x/latest", "loss": 3.5, "loss_text": 1.1, "loss_speech": 2.4})
    md = p.render_markdown()
    assert "step: `100`" in md
    assert "text `1.1`" in md and "speech `2.4`" in md


def test_seed_creates_the_repo_file(paths):
    md = seed_progress_md(paths)
    assert md.exists()
    assert BEGIN in md.read_text(encoding="utf-8")
    assert json.loads(paths.state_json.read_text(encoding="utf-8"))["run"] == "sheng_t3"


def test_notes_are_capped(paths):
    p = Progress(paths)
    for i in range(60):
        p.add_note(f"note {i}")
    assert len(p.load()["notes"]) == 40


def test_tail_events(paths):
    p = Progress(paths)
    p.event("setup", "done")
    p.event("data", "done")
    assert [e["stage"] for e in p.tail_events(2)] == ["setup", "data"]
