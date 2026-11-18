"""``sheng-chatterbox`` command line: the only entry point you need on Colab.

Every subcommand is safe to re-run: it either skips finished work (``all``) or
resumes it (``train``). That is what makes a runtime disconnect cheap - the
state lives in ``<artifacts>/progress/state.json`` (on Google Drive), not in the
notebook.

    sheng-chatterbox env      # pip install chatterbox-tts + deps, mount Drive
    sheng-chatterbox data     # HF dataset -> 16 kHz WAVs -> train/val JSONL
    sheng-chatterbox model    # snapshot ResembleAI/chatterbox into a local dir
    sheng-chatterbox train    # T3 fine-tune (resumes from <save>/latest)
    sheng-chatterbox eval     # synthesize fixed Sheng prompts, optional A/B vs base
    sheng-chatterbox export   # package T3 + base files as a loadable model dir
    sheng-chatterbox status   # what is done, where the checkpoints are, what next
    sheng-chatterbox doctor   # environment diagnosis (GPU, Drive, deps, model, data)
    sheng-chatterbox all      # env -> data -> model -> train -> eval -> export

Deliberately stdlib-only at import time: ``status`` and ``doctor`` must work on
a brand-new Colab VM before anything ML is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import dataset as D
from . import env as E
from . import infer as I
from . import train as T
from .config import DEFAULT_RUN, PROFILES, SMALL_GPU, TrainConfig
from .paths import (
    DATASET_ID,
    DEFAULT_STEPS,
    MODEL_REPO_ID,
    Paths,
    checkpoint_step,
    latest_checkpoint,
    resolve_artifacts,
)
from .progress import STAGES, Progress, seed_progress_md

EPILOG = """\
artifacts root resolution: --artifacts > $SHENG_CHATTERBOX_ARTIFACTS >
/content/drive/MyDrive/chatterbox-sheng (only when Drive is mounted) > <repo>/artifacts.
"""


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--artifacts", default=argparse.SUPPRESS, help="artifacts root (default: auto-detect Drive)")
    common.add_argument("--run", default=argparse.SUPPRESS, help=f"run name (default: {DEFAULT_RUN})")

    p = argparse.ArgumentParser(
        prog="sheng-chatterbox",
        description="Chatterbox T3 fine-tuning for Sheng (Swahili + English) on Colab.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--artifacts", help="artifacts root (default: auto-detect Drive)")
    p.add_argument("--run", default=DEFAULT_RUN, help=f"run name (default: {DEFAULT_RUN})")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, parents=[common])

    env_p = add("env", "install chatterbox-tts + training deps, mount Drive")
    env_p.add_argument("--full-deps", action="store_true", help="official pip install (REPLACES Colab's torch)")
    env_p.add_argument("--reinstall", action="store_true", help="force the pip install even if imports work")
    env_p.add_argument("--skip-drive", action="store_true", help="do not mount Google Drive")

    data_p = add("data", "dataset -> 16 kHz WAVs + train/val manifests")
    data_p.add_argument("--dataset", default=DATASET_ID, help=f"HF dataset id (default: {DATASET_ID})")
    data_p.add_argument("--speaker", default=None, help="keep only this speaker (e.g. abel)")
    data_p.add_argument("--val-clips", type=int, default=4, help="held-out clips (default: 4)")
    data_p.add_argument("--language", default="sw", help="language id written into the manifest (default: sw)")
    data_p.add_argument("--force", action="store_true", help="rebuild even if stats match")

    model_p = add("model", "download ResembleAI/chatterbox into a local directory")
    model_p.add_argument("--force", action="store_true", help="re-download even if the snapshot looks valid")

    train_p = add("train", "T3 fine-tune (resumes automatically)")
    train_p.add_argument("--profile", choices=sorted(PROFILES), help="GPU profile (default: from VRAM)")
    train_p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"total steps (default: {DEFAULT_STEPS})")
    train_p.add_argument("--learning-rate", type=float, default=5e-05)
    train_p.add_argument("--save-interval", type=int, default=100)
    train_p.add_argument("--valid-interval", type=int, default=100)
    train_p.add_argument("--log-interval", type=int, default=10)
    train_p.add_argument("--keep-last", type=int, default=3, help="step dirs kept on disk (default: 3)")
    train_p.add_argument("--max-hours", type=float, default=None, help="stop gracefully before the Colab limit")
    train_p.add_argument("--allow-small-gpu", action="store_true", help="accept a T4 / <20 GB card")
    train_p.add_argument("--fresh", action="store_true", help="archive the previous run directory first")
    train_p.add_argument("--no-resume", action="store_true", help="start from step 0 (keeps old checkpoints)")
    train_p.add_argument("--dry-run", action="store_true", help="one batch forward/backward, then stop")
    train_p.add_argument("--device", default="cuda")

    eval_p = add("eval", "synthesize the eval prompt set with checkpoint(s)")
    eval_p.add_argument("--step", default="latest", help="'latest', 'best' or a step number")
    eval_p.add_argument("--ab", action="store_true", help="also synthesize with the base model")
    eval_p.add_argument("--language", default="sw")
    eval_p.add_argument("--device", default="cuda")

    export_p = add("export", "package T3 + base files as a loadable model dir")
    export_p.add_argument("--step", default="latest", help="'latest' or a step number")
    export_p.add_argument("--no-samples", action="store_true", help="do not copy eval wavs into the package")
    export_p.add_argument("--no-zip", action="store_true", help="skip the .zip archive")

    status_p = add("status", "print what is done and what to run next")
    status_p.add_argument("--json", action="store_true", help="machine-readable output")
    status_p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="steps the run is measured against")

    sub.add_parser("doctor", help="diagnose GPU, Drive, deps, base model and data", parents=[common])

    all_p = add("all", "run every stage, skipping the finished ones")
    all_p.add_argument("--profile", choices=sorted(PROFILES))
    all_p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    all_p.add_argument("--max-hours", type=float, default=None)
    all_p.add_argument("--allow-small-gpu", action="store_true")
    all_p.add_argument("--val-clips", type=int, default=4)
    all_p.add_argument("--resume", action="store_true", help="no-op: finished stages are always skipped")
    all_p.add_argument("--force", action="store_true", help="redo setup/data/model even if marked done")
    all_p.add_argument("--skip-eval", action="store_true")
    all_p.add_argument("--skip-export", action="store_true")
    all_p.add_argument("--ab", action="store_true", help="eval also renders the base model")
    all_p.add_argument("--dry-run", action="store_true")

    return p


# -------------------------------------------------------------------- profile
def pick_profile(*, override: str | None = None, allow_small_gpu: bool = False) -> tuple[str, str | None]:
    """Return (profile, warning). Defaults to the GPU we actually have."""
    if override:
        if override in SMALL_GPU and not allow_small_gpu:
            raise SystemExit(f"profile {override!r} needs --allow-small-gpu (it is slow/VRAM-tight)")
        return override, None

    gpu = E.gpu_info()
    if not gpu.get("cuda"):
        return "cpu", "no CUDA GPU: only `train --dry-run` is meaningful"
    name = str(gpu.get("name", ""))
    vram = float(gpu.get("vram_gb", 0) or 0)
    if "A100" in name or vram >= 38:
        return "a100", None
    if vram >= 20:
        return "l4", None
    if allow_small_gpu:
        return "t4", f"{name} ({vram:.0f} GiB) is small: 2x4 batch, fp16"
    raise SystemExit(
        f"GPU {name} with {vram:.0f} GiB is below the 20 GiB target. "
        "Use an L4/A100 runtime, or pass --allow-small-gpu --profile t4."
    )


# ------------------------------------------------------------------ commands
def cmd_env(paths: Paths, args: argparse.Namespace) -> int:
    progress = Progress(paths)
    if not args.skip_drive:
        E.mount_drive(log=log)
    gpu = E.gpu_info()
    log(f"[env] GPU: {gpu.get('name')} vram={gpu.get('vram_gb')} GiB bf16={gpu.get('bf16')}")

    if args.reinstall or not E.is_installed():
        marker = E.install(full_deps=args.full_deps, force=args.reinstall, log=log)
    else:
        marker = E.marker_state() or {}
        log("[env] already installed (use --reinstall to force)")
        if args.full_deps and not marker.get("full_deps"):
            log("[env] --full-deps requested - reinstalling the official way")
            marker = E.install(full_deps=True, force=True, log=log)

    missing = marker.get("missing") or []
    if missing:
        progress.event("setup", "failed", note=f"missing imports: {missing}")
        log(f"[env] still missing: {missing}")
        return 1

    progress.event(
        "setup",
        "done",
        note=f"chatterbox-tts {E.CHATTERBOX_VERSION} ({'full' if args.full_deps else 'no-deps'} install)",
        probes=marker.get("probes"),
    )
    if args.full_deps:
        log("[env] full install replaced torch -> Runtime > Restart session, then re-run this cell")
    log(f"[env] done. Recommended profile for this GPU: {pick_profile(allow_small_gpu=True)[0]}")
    return 0


def cmd_data(paths: Paths, args: argparse.Namespace) -> int:
    progress = Progress(paths)
    cfg = D.PrepareConfig(
        dataset_id=args.dataset,
        speaker=args.speaker,
        val_clip_count=args.val_clips,
        language=args.language,
        force=args.force,
    )
    info = D.prepare(paths, cfg, log=log)
    progress.event(
        "data",
        "done",
        note=f"{info['train']['clips']} train / {info['val']['clips']} val clips, {info['train']['duration_min']} min",
        dataset=info.get("dataset_id"),
    )
    log(f"[data] report -> {paths.data_report}")
    return 0


def cmd_model(paths: Paths, args: argparse.Namespace) -> int:
    progress = Progress(paths)
    if args.force and paths.pretrained_dir.exists():
        log(f"[model] --force: removing {paths.pretrained_dir}")
        shutil.rmtree(paths.pretrained_dir, ignore_errors=True)
    target = E.download_pretrained(paths, log=log, force=args.force)
    progress.event("model", "done", note=str(target))
    return 0


def _make_config(args: argparse.Namespace, paths: Paths) -> TrainConfig:
    profile, warning = pick_profile(
        override=getattr(args, "profile", None), allow_small_gpu=getattr(args, "allow_small_gpu", False)
    )
    if warning:
        log(f"[train] {warning}")
    return TrainConfig.from_profile(
        profile,
        run=args.run,
        steps=args.steps,
        learning_rate=getattr(args, "learning_rate", 5e-05),
        save_interval=getattr(args, "save_interval", 100),
        valid_interval=getattr(args, "valid_interval", 100),
        log_interval=getattr(args, "log_interval", 10),
        keep_last=getattr(args, "keep_last", 3),
    )


def cmd_train(paths: Paths, args: argparse.Namespace) -> int:
    progress = Progress(paths)
    cfg = _make_config(args, paths)

    if args.fresh:
        _archive_run(paths, cfg.run)
    summary = T.run_training(
        paths,
        cfg,
        progress,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        max_hours=args.max_hours,
        log=log,
    )
    if summary["status"] == "dry-run":
        log("[train] dry-run ok - remove --dry-run to start for real")
        return 0
    if summary["status"] == "done":
        log(f"[train] done at step {summary['step']} ({summary['minutes']} min) -> {summary['checkpoint']}")
        log(f"[train] next: sheng-chatterbox eval --run {args.run} --ab")
        return 0
    if summary["status"] == "partial":
        log(f"[train] stopped at step {summary['step']}/{cfg.steps} - re-run to resume")
        return 0
    log(f"[train] failed: {summary.get('error')}")
    return 1


def cmd_eval(paths: Paths, args: argparse.Namespace) -> int:
    results = I.evaluate(
        paths, args.run, step=args.step, ab=args.ab, language_id=args.language, log=log
    )
    done = sum(len(v) for v in results["prompts"].values())
    log(f"[eval] {done} renders written. Next: sheng-chatterbox export --run {args.run}")
    return 0


def cmd_export(paths: Paths, args: argparse.Namespace) -> int:
    dest = I.export(
        paths,
        args.run,
        step=args.step,
        with_samples=not args.no_samples,
        zip_it=not args.no_zip,
        log=log,
    )
    log(f"[export] model dir: {dest}")
    return 0


def cmd_status(paths: Paths, args: argparse.Namespace) -> int:
    seed_progress_md(paths)  # keeps progress.md present on a fresh clone too
    progress = Progress(paths)
    state = progress.load()
    status, step = T.train_status(paths, args.run, args.steps)
    ckpt = latest_checkpoint(paths, args.run)
    steps_dirs = sorted(
        (p for p in paths.checkpoint_dir(args.run).glob("step_*") if p.is_dir()), key=lambda p: p.name
    ) if paths.checkpoint_dir(args.run).is_dir() else []

    payload: dict[str, Any] = {
        "artifacts": str(paths.artifacts),
        "run": args.run,
        "stages": state["stages"],
        "train_status": status,
        "train_step": step,
        "steps_target": args.steps,
        "latest_checkpoint": str(ckpt) if ckpt else None,
        "checkpoints": [p.name for p in steps_dirs],
        "manifests": {
            "train": paths.train_manifest.exists(),
            "val": paths.val_manifest.exists(),
        },
        "model_downloaded": paths.base_t3_path.exists(),
        "next": progress.next_step(),
        "progress_md": str(paths.repo_progress_md),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    log(f"artifacts : {paths.artifacts}")
    log(f"run       : {args.run}")
    log(f"progress  : {paths.repo_progress_md}")
    log("")
    log("| stage  | status | detail |")
    log("|--------|--------|--------|")
    for stage, (label, _cmd) in STAGES.items():
        info = state["stages"].get(stage, {})
        if stage == "train":
            shown = status if info.get("status") != "running" else "running"
        else:
            shown = info.get("status", "todo")
        log(f"| {stage:<6} | {shown:<7} | {label} |")
    log("")
    log(f"manifests : train={payload['manifests']['train']} val={payload['manifests']['val']}")
    log(f"base model: {'downloaded' if payload['model_downloaded'] else 'missing'} ({paths.pretrained_dir})")
    if steps_dirs:
        log(f"checkpoints: {len(steps_dirs)} ({steps_dirs[0].name} .. {steps_dirs[-1].name})")
        log(f"resume at : step {step} (latest -> {ckpt})")
    else:
        log("checkpoints: none yet")
    log("")
    log(f"next: {payload['next']}")
    return 0


def cmd_doctor(paths: Paths, args: argparse.Namespace) -> int:
    rows = E.doctor(paths)
    log(f"{MODEL_REPO_ID} T3 on {DATASET_ID} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    for label, status, detail in rows:
        tag = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}.get(status, status)
        log(f"[{tag}] {label:<28} {detail}")
    log("")
    failures = [label for label, status, _ in rows if status == "fail"]
    warnings = [label for label, status, _ in rows if status == "warn"]
    if failures:
        log(f"{len(failures)} check(s) failing: {', '.join(failures)}")
        log("Fix with `sheng-chatterbox env`, `data`, `model` as appropriate.")
        return 1
    if warnings:
        log(f"{len(warnings)} warning(s): {', '.join(warnings)}")
    log("All critical checks passed - `sheng-chatterbox all --resume` is safe to run.")
    return 0


def _archive_run(paths: Paths, run: str) -> None:
    src = paths.checkpoint_dir(run)
    if not src.exists():
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = src.with_name(f"{run}.archived-{stamp}")
    log(f"[train] --fresh: moving {src.name} -> {dst.name}")
    src.rename(dst)


def _setup_ready(paths: Paths) -> bool:
    """Is the *VM-local* install still valid?

    ``setup`` is the one stage whose state lives in the VM, not on Drive: a new
    Colab VM has the repo (on Drive) but not the installed packages. The marker
    in /tmp plus an import check is what tells them apart.
    """
    return E.is_installed()


def _stage_already_done(paths: Paths, args: argparse.Namespace, stage: str, progress: Progress) -> bool:
    if args.force:
        return False
    if stage == "setup":
        return _setup_ready(paths)
    if stage == "train":
        status, step = T.train_status(paths, args.run, args.steps)
        return status == "done"
    return progress.is_done(stage)


def cmd_all(paths: Paths, args: argparse.Namespace) -> int:
    progress = Progress(paths)
    started = time.time()

    def run_stage(stage: str, fn: Callable[[], int]) -> None:
        if _stage_already_done(paths, args, stage, progress):
            log(f"[all] {stage}: already done, skipping")
            return
        log(f"\n[all] ===== {stage} =====")
        code = fn()
        if code != 0:
            raise SystemExit(f"stage {stage} failed (exit {code}); see progress.md")

    run_stage("setup", lambda: cmd_env(paths, _env_args(args)))
    run_stage("data", lambda: cmd_data(paths, _data_args(args)))
    run_stage("model", lambda: cmd_model(paths, _model_args(args)))
    run_stage("train", lambda: cmd_train(paths, _train_args(args)))
    if not args.skip_eval:
        run_stage("eval", lambda: cmd_eval(paths, _eval_args(args)))
    if not args.skip_export:
        run_stage("export", lambda: cmd_export(paths, _export_args(args)))

    log(f"\n[all] finished in {(time.time() - started) / 60:.1f} min")
    log(f"[all] {progress.next_step()}")
    log(f"[all] progress -> {paths.repo_progress_md}")
    return 0


# Thin adapters so `all` can reuse the single-stage commands.
def _env_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(run=args.run, full_deps=False, reinstall=False, skip_drive=False)


def _data_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        run=args.run, dataset=DATASET_ID, speaker=None, val_clips=args.val_clips, language="sw", force=False
    )


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(run=args.run, force=False)


def _train_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        run=args.run,
        profile=args.profile,
        steps=args.steps,
        learning_rate=5e-05,
        save_interval=100,
        valid_interval=100,
        log_interval=10,
        keep_last=3,
        max_hours=args.max_hours,
        allow_small_gpu=args.allow_small_gpu,
        fresh=False,
        no_resume=False,
        dry_run=args.dry_run,
        device="cuda",
    )


def _eval_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(run=args.run, step="latest", ab=getattr(args, "ab", False), language="sw", device="cuda")


def _export_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(run=args.run, step="latest", no_samples=False, no_zip=False)


COMMANDS: dict[str, Callable[[Paths, argparse.Namespace], int]] = {
    "env": cmd_env,
    "data": cmd_data,
    "model": cmd_model,
    "train": cmd_train,
    "eval": cmd_eval,
    "export": cmd_export,
    "status": cmd_status,
    "doctor": cmd_doctor,
    "all": cmd_all,
}


def _ensure_artifacts_root(args: argparse.Namespace) -> None:
    """On Colab, mount Drive *before* resolving paths.

    Resolving first would silently fall back to ``<repo>/artifacts`` on the
    ephemeral VM - i.e. the whole point of the resume story is lost. ``doctor``
    is excluded so it can report the real (unmounted) state.
    """
    if args.command == "doctor" or args.artifacts or os.environ.get("SHENG_CHATTERBOX_ARTIFACTS"):
        return
    if Path("/content/drive/MyDrive").is_dir() or not Path("/content").is_dir():
        return
    log("[artifacts] Google Drive not mounted - mounting it so checkpoints survive a disconnect")
    E.mount_drive(log=log)


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_artifacts_root(args)
    paths = Paths(resolve_artifacts(args.artifacts)).ensure()
    if Path("/content").is_dir() and not str(paths.artifacts).startswith("/content/drive"):
        log(f"[artifacts] WARNING: {paths.artifacts} is on the ephemeral VM - a disconnect loses it")
    if not (args.command == "status" and getattr(args, "json", False)):
        log(f"# sheng-chatterbox | artifacts={paths.artifacts} | run={args.run}")
    try:
        return COMMANDS[args.command](paths, args)
    except KeyboardInterrupt:
        log("\ninterrupted - state is saved, re-run the same command to resume")
        return 130
    except (RuntimeError, OSError, ValueError, FileNotFoundError) as exc:
        log(f"error: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
