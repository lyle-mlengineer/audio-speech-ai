"""Driver for the training stage: prerequisites, subprocess, heartbeat, stop guard.

The heavy lifting lives in ``sheng_chatterbox.train_t3`` (argparse-only so it can
also be launched by hand). This module exists for the things a Colab session
needs and a bare script does not:

* verify the run *can* start (torch, chatterbox, manifests, weights) and say
  exactly what to fix instead of dying 200 lines into a stack trace;
* stream the trainer's log to the notebook *and* to
  ``artifacts/logs/<run>/train.log`` (the file survives ``!`` shell restarts);
* write a progress heartbeat every ~30 s so a dead session is obvious from
  ``progress.md``;
* ``--max-hours``: SIGINT the trainer before the Colab limit, let it checkpoint,
  then report the partial state (re-run with ``--resume`` to continue).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import TrainConfig
from .paths import Paths, latest_checkpoint

STEP_RE = re.compile(r"step=(\d+)\s+loss=([0-9.]+)")
VAL_RE = re.compile(r"val step=(\d+)\s+val_loss=([0-9.]+)")

HEARTBEAT_SEC = 30.0
POLL_SEC = 1.0


class MissingPrerequisites(RuntimeError):
    """Raised with a human-readable list of what to run first."""


def check_prerequisites(paths: Paths, cfg: TrainConfig) -> None:
    problems: list[str] = []

    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        problems.append(f"torch not importable ({exc}) -> run `sheng-chatterbox env`")
    else:
        try:
            import chatterbox  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            problems.append(f"chatterbox not importable ({exc}) -> run `sheng-chatterbox env`")

    if not paths.train_manifest.exists():
        problems.append(f"missing train manifest {paths.train_manifest} -> run `sheng-chatterbox data`")
    missing_files = [
        name for name in ("ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", cfg_variant_file(cfg))
        if not (paths.pretrained_dir / name).exists()
    ]
    if missing_files:
        problems.append(
            f"missing model files {missing_files} in {paths.pretrained_dir} -> run `sheng-chatterbox model`"
        )

    if problems:
        raise MissingPrerequisites("\n".join(f"- {p}" for p in problems))


def cfg_variant_file(cfg: TrainConfig) -> str:
    return "t3_mtl23ls_v2.safetensors" if cfg.model_variant == "mtl" else "t3_cfg.safetensors"


def _trainer_command(paths: Paths, cfg: TrainConfig, *, resume: bool, dry_run: bool) -> list[str]:
    argv = [sys.executable, "-m", "sheng_chatterbox.train_t3", *cfg.trainer_args(paths)]
    argv += ["--resume", "auto" if resume else "none"]
    if dry_run:
        argv.append("--dry-run")
    return argv


def tail(path: Path, from_byte: int = 0) -> tuple[list[str], int]:
    """New complete lines since ``from_byte`` plus the new offset."""
    try:
        with path.open("rb") as f:
            f.seek(from_byte)
            chunk = f.read()
    except FileNotFoundError:
        return [], from_byte
    if not chunk:
        return [], from_byte
    text = chunk.decode("utf-8", errors="replace")
    if not text.endswith("\n"):
        cut = text.rfind("\n") + 1
        text, chunk = text[:cut], chunk[:cut]
    offset = from_byte + len(chunk)
    return [ln for ln in text.splitlines() if ln.strip()], offset


def run_training(
    paths: Paths,
    cfg: TrainConfig,
    progress=None,
    *,
    resume: bool = True,
    dry_run: bool = False,
    max_hours: float | None = None,
    log=print,
) -> dict[str, Any]:
    """Launch the trainer, stream its log, and return a summary dict."""
    check_prerequisites(paths, cfg)
    cfg.validate()
    paths.ensure()
    cfg_yaml = cfg.write_yaml(paths)

    log_path = paths.train_log(cfg.run)
    console_path = paths.console_log(cfg.run)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)

    argv = _trainer_command(paths, cfg, resume=resume, dry_run=dry_run)
    log(f"[train] config  -> {cfg_yaml}")
    log(f"[train] log     -> {log_path}")
    log(f"[train] command -> {' '.join(argv)}")
    if progress is not None:
        progress.set_state(run=cfg.run, checkpoint_root=str(paths.checkpoint_dir(cfg.run)))
        progress.event("train", "running", note=f"steps={cfg.steps} profile={cfg.profile}")

    started = time.time()
    log_f = log_path.open("ab")
    proc = subprocess.Popen(
        argv,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}),
    )

    offset = 0
    last_beat = 0.0
    step = 0
    last_loss: float | None = None
    last_val: float | None = None
    stop_sent = False
    stop_at: float | None = None

    try:
        while True:
            lines, offset = tail(log_path, offset)
            for line in lines:
                print(line, flush=True)
                with console_path.open("a", encoding="utf-8") as cf:
                    cf.write(line + "\n")
                m = STEP_RE.search(line)
                if m:
                    step, last_loss = int(m.group(1)), float(m.group(2))
                mv = VAL_RE.search(line)
                if mv:
                    last_val = float(mv.group(2))

            now = time.time()
            if progress is not None and now - last_beat >= HEARTBEAT_SEC:
                last_beat = now
                progress.heartbeat(
                    cfg.run,
                    step,
                    note=lines[-1] if lines else "",
                    **(
                        {"checkpoint": {"step": step, "path": str(paths.latest_dir(cfg.run)), "loss": last_loss}}
                        if step
                        else {}
                    ),
                )

            if max_hours and not stop_sent and (now - started) / 3600.0 >= max_hours:
                stop_sent, stop_at = True, now
                log(f"[train] --max-hours {max_hours} reached - asking the trainer to checkpoint and exit")
                _send_stop(proc)

            if stop_sent and stop_at and now - stop_at > 180:
                log("[train] trainer ignored SIGINT for 3 min - killing it")
                proc.kill()
                break

            if proc.poll() is not None:
                break
            time.sleep(POLL_SEC)
    finally:
        log_f.close()
        # Drain whatever the trainer wrote after the last poll.
        lines, offset = tail(log_path, offset)
        for line in lines:
            print(line, flush=True)
            with console_path.open("a", encoding="utf-8") as cf:
                cf.write(line + "\n")

    returncode = proc.wait()
    summary = _final_summary(paths, cfg, returncode, step, last_loss, last_val, started, dry_run)

    if progress is not None:
        if summary["status"] == "done":
            progress.event(
                "train",
                "done",
                note=f"step {summary['step']}/{cfg.steps}, loss {summary['loss']}, {summary['minutes']} min",
                **{"checkpoint": {"step": summary["step"], "path": str(paths.latest_dir(cfg.run))}},
            )
        elif summary["status"] == "partial":
            progress.event(
                "train",
                "partial",
                note=f"stopped at step {summary['step']}/{cfg.steps} - re-run with --resume",
                **{"checkpoint": {"step": summary["step"], "path": str(paths.latest_dir(cfg.run))}},
            )
        else:
            progress.event("train", "failed", note=summary.get("error", "see train.log"))
    return summary


def _send_stop(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except Exception:  # noqa: BLE001 - fall back to terminate below
        proc.terminate()


def _final_summary(
    paths: Paths,
    cfg: TrainConfig,
    returncode: int,
    step: int,
    loss: float | None,
    val: float | None,
    started: float,
    dry_run: bool,
) -> dict[str, Any]:
    ckpt = latest_checkpoint(paths, cfg.run)
    saved_step = None
    if ckpt is not None:
        try:
            meta = json.loads((ckpt / "training_state.json").read_text(encoding="utf-8"))
            saved_step = int(meta.get("step", 0))
            metrics = meta.get("metrics") or {}
            loss = metrics.get("loss", loss)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    reached = saved_step if saved_step is not None else step
    if dry_run:
        status = "dry-run"
    elif returncode == 0 and reached >= cfg.steps:
        status = "done"
    elif reached > 0:
        status = "partial"
    else:
        status = "failed"

    return {
        "status": status,
        "returncode": returncode,
        "step": reached,
        "steps_planned": cfg.steps,
        "loss": loss,
        "val_loss": val,
        "checkpoint": str(ckpt) if ckpt else None,
        "minutes": round((time.time() - started) / 60, 2),
        "error": None if returncode == 0 else f"trainer exited with code {returncode}, see {paths.train_log(cfg.run)}",
    }


def train_status(paths: Paths, run: str, steps: int) -> tuple[str, int | None]:
    """('done'|'partial'|'todo', step) for the requested ``steps``."""
    ckpt = latest_checkpoint(paths, run)
    if ckpt is None:
        return "todo", None
    try:
        meta = json.loads((ckpt / "training_state.json").read_text(encoding="utf-8"))
        step = int(meta.get("step", 0))
    except (OSError, json.JSONDecodeError, ValueError):
        step = 0
    return ("done" if step >= steps else "partial"), step
