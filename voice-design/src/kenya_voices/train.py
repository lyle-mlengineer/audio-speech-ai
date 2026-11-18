"""Train-stage driver: prereqs, GPU profile, YAML, resume, watchdog, heartbeat.

Why a wrapper at all:

* ``train_voxcpm_finetune.py`` auto-resumes from ``<save_path>/latest`` but has no
  idea about Colab sessions, Drive, or our dataset. This module makes the run
  repeatable after a disconnect and keeps ``progress.md``/``state.json`` current
  *while* training runs (heartbeat), so a dead session still leaves a usable
  record of how far it got.
* ``--max-hours`` sends SIGINT to the trainer just before the Colab limit; the
  trainer's own signal handler saves a checkpoint, so nothing is lost.
* The GPU profile is picked from real VRAM (see config.py).

Resume contract: training always continues from ``<save_path>/latest`` (or the
highest ``step_*`` folder). ``--fresh`` archives the previous run directory
instead of deleting it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import dataset as D
from . import env as E
from .config import DEFAULT_STEPS, PROFILES, TrainConfig, profile_summary, write_config
from .paths import DEFAULT_RUN, Paths
from .progress import Progress

_STEP_DIR = re.compile(r"^step_(\d+)$")
_LOSS_LINE = re.compile(r"loss/diff[^0-9\-]*(-?\d+\.?\d*)")


# ----------------------------------------------------------------- helpers
def checkpoints(save_dir: Path) -> list[tuple[int, Path]]:
    """``step_*`` checkpoint folders, ascending by step (``latest`` excluded)."""
    if not save_dir.is_dir():
        return []
    out = []
    for child in save_dir.iterdir():
        m = _STEP_DIR.match(child.name)
        if m and child.is_dir():
            out.append((int(m.group(1)), child))
    return sorted(out)


def latest_checkpoint(save_dir: Path) -> Path | None:
    latest = save_dir / "latest"
    if latest.is_dir() and any(latest.iterdir()):
        return latest
    ckpts = checkpoints(save_dir)
    return ckpts[-1][1] if ckpts else None


def resume_step(save_dir: Path) -> int:
    latest = latest_checkpoint(save_dir)
    if latest is None:
        return 0
    try:
        state = json.loads((latest / "training_state.json").read_text(encoding="utf-8"))
        return int(state.get("step", 0))
    except (OSError, json.JSONDecodeError, ValueError):
        return 0


def archive_run(save_dir: Path) -> Path | None:
    if not save_dir.exists() or not any(save_dir.iterdir()):
        return None
    dest = save_dir.parent / f"{save_dir.name}_archive_{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.move(str(save_dir), str(dest))
    return dest


def _parse_override(raw: str) -> tuple[str, Any]:
    key, _, value = raw.partition("=")
    if not _:
        raise ValueError(f"--set expects key=value, got {raw!r}")
    value = value.strip()
    try:
        return key.strip(), json.loads(value)
    except json.JSONDecodeError:
        return key.strip(), value


@dataclass
class TrainOptions:
    run: str = DEFAULT_RUN
    profile: str | None = None  # None -> auto from GPU
    steps: int = DEFAULT_STEPS
    learning_rate: float = 1e-4
    lora_r: int = 32
    lora_alpha: int | None = None  # None -> same as r
    save_interval: int = 100
    valid_interval: int = 100
    log_interval: int = 10
    allow_small_gpu: bool = False
    fresh: bool = False
    max_hours: float | None = None
    skip_prereqs: bool = False
    dry_run: bool = False
    install_deps: bool = True
    data: Any = None  # dataset.PrepareConfig used when data is (re)built
    overrides: list[str] = field(default_factory=list)


# -------------------------------------------------------------------- main
def train(paths: Paths, opts: TrainOptions, *, log=print) -> dict[str, Any]:
    paths.ensure()
    progress = Progress(paths)

    save_dir = paths.checkpoint_dir(opts.run)
    if opts.fresh and save_dir.exists() and any(save_dir.iterdir()):
        archived = archive_run(save_dir)
        log(f"[train] --fresh: archived previous run to {archived}")
        progress.add_note(f"archived previous {opts.run} run to {archived}")

    # --dry-run renders the YAML only: no downloads, no GPU required.
    if opts.dry_run:
        gpu = E.gpu_info()
        profile = opts.profile or (gpu.profile if gpu.available else "l4")
        log("[train] --dry-run: skipping prereqs and GPU checks")
    else:
        gpu = E.check_gpu(allow_small=opts.allow_small_gpu)
        profile = opts.profile or gpu.profile
    if profile not in PROFILES:
        raise SystemExit(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    if not opts.dry_run and profile == "t4" and not opts.allow_small_gpu and not gpu.bf16:
        raise SystemExit(
            f"{gpu.name} has no native bf16 and needs the reduced profile. "
            "Re-run with --allow-small-gpu (or use an L4/A100)."
        )

    cfg = TrainConfig(
        run=opts.run,
        profile=profile,
        steps=opts.steps,
        learning_rate=opts.learning_rate,
        lora_r=opts.lora_r,
        lora_alpha=opts.lora_alpha if opts.lora_alpha is not None else opts.lora_r,
        save_interval=opts.save_interval,
        valid_interval=opts.valid_interval,
        log_interval=opts.log_interval,
        overrides={},
    )
    for raw in opts.overrides:
        key, value = _parse_override(raw)
        cfg.overrides[key] = value

    cfg_path = write_config(paths, cfg)
    start_step = resume_step(save_dir)
    log(f"[train] {profile_summary(cfg)}")
    log(f"[train] config -> {cfg_path}")
    log(f"[train] resume from step {start_step}" if start_step else "[train] starting from step 0")

    if start_step >= opts.steps:
        note = f"already at step {start_step} >= target {opts.steps}; pass a larger --steps to continue"
        log(f"[train] {note}")
        progress.event("train", "done", note=note, step=start_step, path=str(latest_checkpoint(save_dir)))
        return {"ok": True, "step": start_step, "config": str(cfg_path), "skipped": True}

    if opts.dry_run:
        log("[train] --dry-run: not launching the trainer")
        return {"ok": True, "step": start_step, "config": str(cfg_path), "dry_run": True}

    if not opts.skip_prereqs:
        run_prereqs(paths, opts, progress, log=log)
    else:
        _assert_prereqs(paths)

    progress.event("train", "running", note=profile_summary(cfg), step=start_step, run=opts.run)
    result = _launch(paths, opts, progress, cfg_path, gpu, summary=profile_summary(cfg), log=log)

    final_step = resume_step(save_dir)
    ckpt = latest_checkpoint(save_dir)
    status = "done" if result["returncode"] == 0 else "partial"
    note = result.get("note") or f"returncode={result['returncode']} step={final_step}"
    progress.event(
        "train",
        status,
        note=note,
        step=final_step,
        path=str(ckpt) if ckpt else None,
        returncode=result["returncode"],
    )
    if ckpt:
        progress.set_state(checkpoint={"step": final_step, "path": str(ckpt)})
    result.update({"ok": result["returncode"] == 0, "step": final_step, "checkpoint": str(ckpt) if ckpt else None})
    return result


def _launch(
    paths: Paths,
    opts: TrainOptions,
    progress: Progress,
    cfg_path: Path,
    gpu: E.GpuInfo,
    *,
    summary: str = "",
    log=print,
) -> dict[str, Any]:
    cmd = [sys.executable, str(paths.train_script), "--config_path", str(cfg_path)]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTHONPATH", str(paths.voxcpm_repo / "src"))
    log_path = paths.console_log(opts.run)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The trainer appends its own loss/metric lines to <save_path>/train.log.
    trainer_log = paths.trainer_log(opts.run)

    log(f"[train] $ {' '.join(cmd)}   (cwd={paths.voxcpm_repo})")
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(paths.voxcpm_repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    stop = threading.Event()
    shared: dict[str, Any] = {"last_line": "", "step": resume_step(paths.checkpoint_dir(opts.run))}

    def _heartbeat() -> None:
        while not stop.wait(60):
            step = resume_step(paths.checkpoint_dir(opts.run))
            if step:
                shared["step"] = step
            note = shared["last_line"]
            if trainer_log.exists():
                try:
                    tail = trainer_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-1][-300:]
                    if tail.strip():
                        note = tail.strip()
                except OSError:
                    pass
            try:
                progress.heartbeat(opts.run, shared["step"], note=note, gpu=gpu.as_dict())
            except Exception as exc:  # noqa: BLE001 - never kill training for a progress write
                log(f"[train] heartbeat failed (ignored): {exc}")

    def _deadline() -> None:
        if not opts.max_hours:
            return
        seconds = max(30.0, opts.max_hours * 3600 - started)
        if stop.wait(seconds):
            return
        log(f"[train] --max-hours {opts.max_hours} reached: asking the trainer to checkpoint and stop")
        progress.add_note(f"time budget {opts.max_hours}h reached; sent interrupt at step {shared['step']}")
        try:
            proc.send_signal(signal.SIGINT)
        except Exception as exc:  # noqa: BLE001
            log(f"[train] could not signal trainer: {exc}")

    threads = [threading.Thread(target=_heartbeat, daemon=True), threading.Thread(target=_deadline, daemon=True)]
    for t in threads:
        t.start()

    note = ""
    returncode = 1
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== run {time.strftime('%Y-%m-%d %H:%M:%S')} | {summary or profile_summary(TrainConfig(run=opts.run))} =====\n")
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                fh.write(line + "\n")
                fh.flush()
                if line.strip():
                    shared["last_line"] = line.strip()[:300]
                    print(line)
            returncode = proc.wait()
        except KeyboardInterrupt:
            note = "interrupted by user; trainer should have saved a checkpoint"
            log("[train] KeyboardInterrupt - forwarding SIGINT to the trainer")
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:  # noqa: BLE001
                pass
            try:
                returncode = proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = -9
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=2)

    if returncode != 0 and not note:
        note = _diagnose(log_path)
    log(f"[train] finished with returncode {returncode} after {time.time() - started:.0f}s")
    return {"returncode": returncode, "note": note, "console_log": str(log_path)}


def _diagnose(log_path: Path) -> str:
    """Turn the usual Colab failures into something actionable."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")[-20000:].lower()
    except OSError:
        return "trainer exited non-zero (no console log)"
    if "out of memory" in text or "cuda error: out of memory" in text:
        return (
            "OOM: rerun with the next profile down (--profile t4) or --set batch_size=1 "
            "--set max_batch_tokens=4096"
        )
    if "sample_rate mismatch" in text:
        return "sample_rate mismatch: VoxCPM2 needs sample_rate=16000 (do not edit the rendered YAML)"
    if "modulenotfounderror" in text:
        return "missing dependency: run `kenya-voices env` (or `pip install -e .` in the artifacts repo)"
    return "trainer exited non-zero - see the console log above"


# ----------------------------------------------------------------- prereqs
def run_prereqs(paths: Paths, opts: TrainOptions, progress: Progress | None = None, *, log=print) -> None:
    progress = progress or Progress(paths)
    E.ensure_repo(paths, log=log)
    progress.event("setup", "done", note=f"voxcpm installed ({E.git_rev(paths.voxcpm_repo)})")

    D.prepare(paths, opts.data or D.PrepareConfig(), log=log)
    stats = json.loads(paths.data_stats.read_text(encoding="utf-8"))
    progress.event(
        "data",
        "done",
        note=f"{stats['train']['clips']} train / {stats['val']['clips']} val clips",
        dataset=stats.get("dataset_id"),
    )

    E.ensure_model(paths, log=log)
    progress.event("model", "done", note=str(paths.model_dir))
    _assert_prereqs(paths)


def _assert_prereqs(paths: Paths) -> None:
    missing = []
    if not paths.train_manifest.exists():
        missing.append(f"train manifest {paths.train_manifest}")
    if not paths.base_config_json.exists():
        missing.append(f"base model {paths.model_dir}")
    if not paths.train_script.exists():
        missing.append(f"VoxCPM repo {paths.voxcpm_repo}")
    if missing:
        raise SystemExit(
            "missing prerequisites:\n  - " + "\n  - ".join(missing) + "\nRun `kenya-voices all` (or `env`, `data`, `model`)."
        )


def status(paths: Paths, run: str = DEFAULT_RUN) -> dict[str, Any]:
    progress = Progress(paths)
    state = progress.load()
    save_dir = paths.checkpoint_dir(run)
    ckpts = checkpoints(save_dir)
    return {
        "artifacts": str(paths.artifacts),
        "run": run,
        "stages": {k: v.get("status", "todo") for k, v in state.get("stages", {}).items()},
        "current_step": state.get("current_step"),
        "checkpoints": [{"step": s, "path": str(p)} for s, p in ckpts],
        "latest": str(latest_checkpoint(save_dir)) if latest_checkpoint(save_dir) else None,
        "resume_step": resume_step(save_dir),
        "manifests": {
            "train": paths.train_manifest.exists(),
            "val": paths.val_manifest.exists(),
        },
        "model_downloaded": paths.base_config_json.exists(),
        "voxcpm_repo": paths.train_script.exists(),
        "train_log": str(paths.train_log(run)),
        "trainer_log": str(paths.trainer_log(run)),
    }
