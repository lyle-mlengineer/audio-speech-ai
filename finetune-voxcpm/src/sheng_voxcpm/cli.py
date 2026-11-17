"""``sheng-voxcpm`` command line: the only entry point you need on Colab.

Every subcommand is safe to re-run: it either skips finished work (``all``) or
resumes it (``train``). That is what makes a runtime disconnect cheap - the
state lives in ``<artifacts>/progress/state.json`` (on Google Drive), not in the
notebook.

    sheng-voxcpm env      # clone + install OpenBMB/VoxCPM + curated deps
    sheng-voxcpm data     # HF dataset -> 16 kHz WAVs -> train/val JSONL
    sheng-voxcpm model    # snapshot openbmb/VoxCPM2 into a local dir
    sheng-voxcpm train    # LoRA fine-tune (resumes from <save_path>/latest)
    sheng-voxcpm eval     # synthesize fixed Sheng prompts, optional A/B vs base
    sheng-voxcpm export   # package the adapter with a portable lora_config.json
    sheng-voxcpm status   # what is done, where the checkpoints are, what next
    sheng-voxcpm doctor   # environment diagnosis (GPU, Drive, deps, model, data)
    sheng-voxcpm all      # env -> data -> model -> train -> eval -> export

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
from .config import DEFAULT_STEPS, LORA_R, PROFILES
from .paths import BASE_MODEL_ID, DATASET_ID, DEFAULT_RUN, Paths
from .progress import STAGES, Progress, seed_progress_md

EPILOG = """\
artifacts root resolution: --artifacts > $SHENG_ARTIFACTS > /content/drive/MyDrive/voxcpm2-sheng
(only when Drive is mounted) > <repo>/artifacts.
"""


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    # Also accepted *after* the subcommand (`sheng-voxcpm status --artifacts ...`).
    # SUPPRESS keeps the main-parser value when the flag is not repeated.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--artifacts", default=argparse.SUPPRESS, help="artifacts root (default: auto-detect Drive)")
    common.add_argument("--run", default=argparse.SUPPRESS, help=f"run name (default: {DEFAULT_RUN})")

    p = argparse.ArgumentParser(
        prog="sheng-voxcpm",
        description="VoxCPM2 LoRA fine-tuning for Sheng (Swahili + English) on Colab.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--artifacts", help="artifacts root (default: auto-detect Drive)")
    p.add_argument("--run", default=DEFAULT_RUN, help=f"run name (default: {DEFAULT_RUN})")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, parents=[common])

    env_p = add("env", "clone + install the VoxCPM repo and its deps")
    env_p.add_argument("--ref", default="main", help="git ref of OpenBMB/VoxCPM")
    env_p.add_argument("--update", action="store_true", help="git fetch --all --tags first")
    env_p.add_argument("--full-deps", action="store_true", help="official `pip install -e .` (may replace torch)")
    env_p.add_argument("--reinstall", action="store_true", help="force the pip install even if the marker matches")
    env_p.add_argument("--skip-drive", action="store_true", help="do not mount Google Drive")

    data_p = add("data", "dataset -> 16 kHz WAVs + train/val manifests")
    data_p.add_argument("--dataset", default=DATASET_ID, help=f"HF dataset id (default: {DATASET_ID})")
    data_p.add_argument("--speaker", default=None, help="keep only this speaker")
    data_p.add_argument("--val-clips", type=int, default=4, help="held-out clips (default: 4)")
    data_p.add_argument("--ref-audio-fraction", type=float, default=0.0, help="share of clips with ref_audio (0-1)")
    data_p.add_argument("--force", action="store_true", help="rebuild even if stats match")

    model_p = add("model", "download openbmb/VoxCPM2 into a local directory")
    model_p.add_argument("--revision", default=None)
    model_p.add_argument("--force", action="store_true", help="re-download even if the snapshot looks valid")

    train_p = add("train", "LoRA fine-tuning (resumes automatically)")
    train_p.add_argument("--profile", choices=sorted(PROFILES), help="GPU profile (default: from VRAM)")
    train_p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"total steps (default: {DEFAULT_STEPS})")
    train_p.add_argument("--learning-rate", type=float, default=1e-4)
    train_p.add_argument("--lora-r", type=int, default=LORA_R)
    train_p.add_argument("--lora-alpha", type=int, default=None, help="default: same as --lora-r")
    train_p.add_argument("--save-interval", type=int, default=100)
    train_p.add_argument("--valid-interval", type=int, default=100)
    train_p.add_argument("--log-interval", type=int, default=10)
    train_p.add_argument("--max-hours", type=float, default=None, help="stop gracefully before the Colab limit")
    train_p.add_argument("--allow-small-gpu", action="store_true", help="accept a T4 / <20 GB card")
    train_p.add_argument("--fresh", action="store_true", help="archive the previous run directory first")
    train_p.add_argument("--resume", action="store_true", help="no-op (resuming is already the default)")
    train_p.add_argument("--skip-prereqs", action="store_true", help="assume env/data/model are ready")
    train_p.add_argument("--dry-run", action="store_true", help="render the YAML and stop")
    train_p.add_argument("--ref-audio-fraction", type=float, default=0.0)
    train_p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="extra YAML key (repeatable)"
    )

    eval_p = add("eval", "synthesize the eval prompt set with checkpoint(s)")
    eval_p.add_argument("--steps", type=int, nargs="*", default=None, help="checkpoint step(s); default: latest")
    eval_p.add_argument("--ab", action="store_true", help="also synthesize with LoRA disabled")
    eval_p.add_argument("--limit", type=int, default=None, help="only the first N eval prompts")
    eval_p.add_argument("--optimize", action="store_true", help="torch.compile (slower start, faster clips)")
    eval_p.add_argument("--device", default="cuda")

    export_p = add("export", "package the LoRA adapter + eval samples")
    export_p.add_argument("--step", type=int, default=None, help="default: latest checkpoint")
    export_p.add_argument("--no-samples", action="store_true", help="do not copy eval wavs into the package")

    status_p = add("status", "print what is done and what to run next")
    status_p.add_argument("--json", action="store_true", help="machine-readable output")

    sub.add_parser("doctor", help="diagnose GPU, Drive, deps, base model and data", parents=[common])

    all_p = add("all", "run every stage, skipping the finished ones")
    all_p.add_argument("--profile", choices=sorted(PROFILES))
    all_p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    all_p.add_argument("--max-hours", type=float, default=None)
    all_p.add_argument("--allow-small-gpu", action="store_true")
    all_p.add_argument("--val-clips", type=int, default=4)
    all_p.add_argument("--ref-audio-fraction", type=float, default=0.0)
    all_p.add_argument("--resume", action="store_true", help="no-op: finished stages are always skipped")
    all_p.add_argument("--force", action="store_true", help="redo setup/data/model even if marked done")
    all_p.add_argument("--skip-eval", action="store_true")
    all_p.add_argument("--skip-export", action="store_true")
    all_p.add_argument("--dry-run", action="store_true")

    return p


# ------------------------------------------------------------------ commands
def cmd_env(paths: Paths, args: argparse.Namespace) -> int:
    progress = Progress(paths)
    if not args.skip_drive:
        E.mount_drive(log=log)
    gpu = E.gpu_info()
    log(f"[env] GPU: {gpu.summary()}")
    provenance = E.ensure_repo(
        paths,
        ref=args.ref,
        update=args.update,
        full_deps=args.full_deps,
        reinstall=args.reinstall,
        log=log,
    )
    progress.event("setup", "done", note=f"voxcpm {provenance['rev']} ({provenance['mode']} deps)", **provenance)
    log(f"[env] done. Recommended training profile for this GPU: {gpu.profile}")
    return 0


def cmd_data(paths: Paths, args: argparse.Namespace) -> int:
    progress = Progress(paths)
    cfg = D.PrepareConfig(
        dataset_id=args.dataset,
        speaker=args.speaker,
        val_clip_count=args.val_clips,
        ref_audio_fraction=args.ref_audio_fraction,
        force=args.force,
    )
    stats = D.prepare(paths, cfg, log=log)
    progress.event(
        "data",
        "done",
        note=f"{stats['train']['clips']} train / {stats['val']['clips']} val clips, "
        f"{stats['train']['duration_min']} min",
        dataset=stats.get("dataset_id"),
    )
    log(f"[data] report -> {paths.data_report}")
    return 0


def cmd_model(paths: Paths, args: argparse.Namespace) -> int:
    progress = Progress(paths)
    if args.force and paths.model_dir.exists():
        log(f"[model] --force: removing {paths.model_dir}")
        shutil.rmtree(paths.model_dir)
    target = E.ensure_model(paths, revision=args.revision, log=log)
    progress.event("model", "done", note=str(target))
    return 0


def _train_options(args: argparse.Namespace) -> T.TrainOptions:
    return T.TrainOptions(
        run=args.run,
        profile=args.profile,
        steps=args.steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        save_interval=args.save_interval,
        valid_interval=args.valid_interval,
        log_interval=args.log_interval,
        allow_small_gpu=args.allow_small_gpu,
        fresh=getattr(args, "fresh", False),
        max_hours=args.max_hours,
        skip_prereqs=getattr(args, "skip_prereqs", False),
        dry_run=getattr(args, "dry_run", False),
        ref_audio_fraction=args.ref_audio_fraction,
        overrides=list(getattr(args, "overrides", []) or []),
    )


def cmd_train(paths: Paths, args: argparse.Namespace) -> int:
    result = T.train(paths, _train_options(args), log=log)
    if result.get("dry_run"):
        log(f"[train] --dry-run: config -> {result['config']} (trainer not launched)")
        return 0
    if result.get("skipped"):
        log(f"[train] already at step {result['step']}; pass --steps {result['step'] + 100} to keep training")
        return 0
    if result.get("ok"):
        log(f"[train] ok at step {result['step']} -> {result.get('checkpoint')}")
        log(f"[train] next: sheng-voxcpm eval --run {args.run}")
        return 0
    log(f"[train] failed: {result.get('note')}")
    return 1


def cmd_eval(paths: Paths, args: argparse.Namespace) -> int:
    summary = I.evaluate(
        paths,
        I.EvalOptions(
            run=args.run,
            steps=args.steps or None,
            ab=args.ab,
            limit=args.limit,
            optimize=args.optimize,
            device=args.device,
        ),
        log=log,
    )
    log(f"[eval] {len(summary['results'])} wavs written. Next: sheng-voxcpm export --run {args.run}")
    return 0


def cmd_export(paths: Paths, args: argparse.Namespace) -> int:
    dest = I.export(
        paths,
        I.ExportOptions(run=args.run, step=args.step, include_samples=not args.no_samples),
        log=log,
    )
    log(f"[export] adapter package: {dest}")
    return 0


def cmd_status(paths: Paths, args: argparse.Namespace) -> int:
    seed_progress_md(paths)  # keeps progress.md present on a fresh clone too
    state = T.status(paths, args.run)
    progress = Progress(paths)
    payload = {
        "artifacts": state["artifacts"],
        "run": state["run"],
        "stages": state["stages"],
        "current_step": state["current_step"],
        "resume_step": state["resume_step"],
        "latest_checkpoint": state["latest"],
        "checkpoints": state["checkpoints"],
        "manifests": state["manifests"],
        "model_downloaded": state["model_downloaded"],
        "voxcpm_repo": state["voxcpm_repo"],
        "next": progress.next_step(),
        "progress_md": str(paths.repo_progress_md),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    log(f"artifacts : {paths.artifacts}")
    log(f"run       : {payload['run']}")
    log(f"progress  : {payload['progress_md']}")
    log("")
    log("| stage  | status | detail |")
    log("|--------|--------|--------|")
    for stage, (label, _cmd) in STAGES.items():
        info = progress.load()["stages"].get(stage, {})
        log(f"| {stage:<6} | {info.get('status', 'todo'):<6} | {label} |")
    log("")
    log(f"manifests : train={payload['manifests']['train']} val={payload['manifests']['val']}")
    log(f"base model: {'downloaded' if payload['model_downloaded'] else 'missing'} ({paths.model_dir})")
    log(f"voxcpm    : {'cloned' if payload['voxcpm_repo'] else 'missing'} ({paths.voxcpm_repo})")
    if payload["checkpoints"]:
        log(f"checkpoints: {len(payload['checkpoints'])} last={payload['latest_checkpoint']}")
        log(f"resume at : step {payload['resume_step']}")
    else:
        log("checkpoints: none yet")
    log("")
    log(f"next: {payload['next']}")
    return 0


def cmd_doctor(paths: Paths, args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []

    py = ".".join(str(v) for v in sys.version_info[:3])
    py_ok = (3, 10) <= sys.version_info[:2] < (3, 13)
    checks.append(("python 3.10-3.12", py_ok, py))

    gpu = E.gpu_info()
    checks.append(("CUDA GPU", gpu.available, gpu.summary()))
    if gpu.available:
        checks.append(
            ("native bf16 (cc >= 8.0)", gpu.bf16, f"cc {gpu.cc:.1f} -> profile {gpu.profile}")
        )
        checks.append((f"VRAM >= {E.MIN_VRAM_GB:.0f} GB", gpu.vram_gb + 0.01 >= E.MIN_VRAM_GB, f"{gpu.vram_gb:.1f} GB"))

    drive = Path("/content/drive/MyDrive")
    on_colab = Path("/content").is_dir()
    checks.append(("Google Drive mounted", drive.is_dir(), str(drive) if drive.is_dir() else "not mounted"))

    try:
        usage = shutil.disk_usage(paths.artifacts)
        free_gb = usage.free / 1024**3
        checks.append(("artifacts writable, >20 GB free", free_gb >= 20, f"{free_gb:.1f} GB free at {paths.artifacts}"))
    except OSError as exc:
        checks.append(("artifacts writable", False, f"{paths.artifacts}: {exc}"))

    try:
        marker = E.marker_path().read_text(encoding="utf-8")
        checks.append(("VoxCPM installed", True, marker.replace("\n", " ")[:120]))
    except OSError:
        checks.append(("VoxCPM installed", False, "run `sheng-voxcpm env`"))

    missing = E.missing_imports()
    checks.append(("required imports", not missing, ", ".join(missing) or "torch, torchaudio, datasets, ..."))

    checks.append(
        ("VoxCPM repo", paths.train_script.exists(), f"{paths.voxcpm_repo} @ {E.git_rev(paths.voxcpm_repo)}")
    )
    arch = E.model_architecture(paths.model_dir)
    checks.append(("base model (voxcpm2)", arch == "voxcpm2", f"{paths.model_dir} architecture={arch}"))

    for label, manifest in (("train manifest", paths.train_manifest), ("val manifest", paths.val_manifest)):
        count = 0
        if manifest.exists():
            with manifest.open("r", encoding="utf-8") as fh:
                count = sum(1 for line in fh if line.strip())
        checks.append((label, count > 0, f"{manifest} ({count} rows)"))

    if paths.data_stats.exists():
        try:
            stats = json.loads(paths.data_stats.read_text(encoding="utf-8"))
            checks.append(
                (
                    "dataset stats",
                    True,
                    f"{stats['train']['duration_min']} min train / {stats['val']['duration_min']} min val",
                )
            )
        except (json.JSONDecodeError, KeyError):
            checks.append(("dataset stats", False, str(paths.data_stats)))

    log(f"{BASE_MODEL_ID} LoRA on {DATASET_ID} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    for label, ok, detail in checks:
        log(f"[{'ok  ' if ok else 'FAIL'}] {label:<28} {detail}")
    log("")
    if on_colab:
        log(f"Colab detected. Recommended profile: {gpu.profile}")
    failures = [label for label, ok, _ in checks if not ok]
    if failures:
        log(f"{len(failures)} check(s) failing: {', '.join(failures)}")
        log("Fix with `sheng-voxcpm env`, `sheng-voxcpm data`, `sheng-voxcpm model` as appropriate.")
        return 1
    log("All checks passed - `sheng-voxcpm all --resume` is safe to run.")
    return 0


def _setup_ready(paths: Paths) -> bool:
    """Is the *VM-local* install still valid?

    ``setup`` is the one stage whose state lives in the VM, not on Drive: a new
    Colab VM has the repo (on Drive) but not the installed packages. The marker
    in ``~/.cache/sheng-voxcpm`` plus an import check is what tells them apart.
    """
    return E.marker_path().exists() and paths.train_script.exists() and not E.missing_imports()


def _stage_already_done(paths: Paths, args: argparse.Namespace, stage: str, progress: Progress) -> bool:
    if args.force:
        return False
    if stage == "setup":
        return _setup_ready(paths)
    if stage == "train":
        # Training is "done" only relative to the requested step count.
        return progress.is_done(stage) and T.resume_step(paths.checkpoint_dir(args.run)) >= args.steps
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
    return argparse.Namespace(
        run=args.run, ref="main", update=False, full_deps=False, reinstall=False, skip_drive=False
    )


def _data_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        run=args.run,
        dataset=DATASET_ID,
        speaker=None,
        val_clips=args.val_clips,
        ref_audio_fraction=args.ref_audio_fraction,
        force=False,
    )


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(run=args.run, revision=None, force=False)


def _train_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        run=args.run,
        profile=args.profile,
        steps=args.steps,
        learning_rate=1e-4,
        lora_r=LORA_R,
        lora_alpha=None,
        save_interval=100,
        valid_interval=100,
        log_interval=10,
        max_hours=args.max_hours,
        allow_small_gpu=args.allow_small_gpu,
        fresh=False,
        resume=True,
        skip_prereqs=True,  # env/data/model just ran
        dry_run=args.dry_run,
        ref_audio_fraction=args.ref_audio_fraction,
        overrides=[],
    )


def _eval_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(run=args.run, steps=None, ab=False, limit=None, optimize=False, device="cuda")


def _export_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(run=args.run, step=None, no_samples=False)


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
    if args.command == "doctor" or args.artifacts or os.environ.get("SHENG_ARTIFACTS"):
        return
    if Path("/content/drive/MyDrive").is_dir() or not Path("/content").is_dir():
        return
    log("[artifacts] Google Drive not mounted - mounting it so checkpoints survive a disconnect")
    E.mount_drive(log=log)


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_artifacts_root(args)
    paths = Paths.resolve(args.artifacts).ensure()
    if Path("/content").is_dir() and not str(paths.artifacts).startswith("/content/drive"):
        log(f"[artifacts] WARNING: {paths.artifacts} is on the ephemeral VM - a disconnect loses it")
    if not (args.command == "status" and getattr(args, "json", False)):
        # Keep `status --json` machine-readable on stdout.
        log(f"# {Path(__file__).resolve().parents[1].name} | artifacts={paths.artifacts} | run={args.run}")
    try:
        return COMMANDS[args.command](paths, args)
    except KeyboardInterrupt:
        log("\ninterrupted - state is saved, re-run the same command to resume")
        return 130
    except (RuntimeError, OSError, ValueError) as exc:
        log(f"error: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
