"""``kenya-voices`` command line: the only entry point you need on Colab.

Every subcommand is safe to re-run: it either skips finished work (``all``) or
resumes it (``train``). That is what makes a runtime disconnect cheap - the state
lives in ``<artifacts>/progress/state.json`` (on Google Drive), not in the
notebook.

    kenya-voices env      # clone + install OpenBMB/VoxCPM + curated deps
    kenya-voices data     # Kenyan voices -> 16 kHz WAVs -> caption manifests
    kenya-voices model    # snapshot openbmb/VoxCPM2 into a local dir
    kenya-voices train    # caption-conditioned LoRA (resumes from <save_path>/latest)
    kenya-voices eval     # design prompts, caption-adherence metrics, optional A/B
    kenya-voices export   # package the adapter + design palette
    kenya-voices design   # synthesize one clip from a description
    kenya-voices status   # what is done, where the checkpoints are, what next
    kenya-voices doctor   # environment diagnosis (GPU, Drive, deps, model, data)
    kenya-voices datasets # available dataset presets
    kenya-voices all      # env -> data -> model -> train -> eval -> export

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
from .paths import BASE_MODEL_ID, DATASET_PRESETS, DEFAULT_DATASETS, DEFAULT_RUN, Paths
from .progress import STAGES, Progress, seed_progress_md

EPILOG = """\
artifacts root resolution: --artifacts > $KENYA_ARTIFACTS > /content/drive/MyDrive/kenya-voices
(only when Drive is mounted) > <repo>/artifacts.
"""


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--artifacts", default=argparse.SUPPRESS, help="artifacts root (default: auto-detect Drive)")
    common.add_argument("--run", default=argparse.SUPPRESS, help=f"run name (default: {DEFAULT_RUN})")

    p = argparse.ArgumentParser(
        prog="kenya-voices",
        description="Caption-conditioned VoxCPM2 LoRA for designing Kenyan voices on Colab.",
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

    data_p = add("data", "Kenyan voices -> 16 kHz WAVs + caption manifests")
    data_p.add_argument("--preset", default=DEFAULT_DATASETS[0], choices=sorted(DATASET_PRESETS))
    data_p.add_argument("--dataset", default="", help="override the preset's HF dataset id")
    data_p.add_argument("--split", default="", help="dataset split (default: preset / train)")
    data_p.add_argument("--speakers", type=int, default=120, help="max speakers to keep (default: 120)")
    data_p.add_argument("--clips-per-speaker", type=int, default=30, help="max clips per speaker (default: 30)")
    data_p.add_argument("--min-clips-per-speaker", type=int, default=6)
    data_p.add_argument("--hours", type=float, default=10.0, help="audio budget in hours (default: 10)")
    data_p.add_argument("--val-speakers", type=float, default=0.1, help="share of speakers held out (default: 0.1)")
    data_p.add_argument("--val-clips-per-speaker", type=int, default=3)
    data_p.add_argument("--caption-variants", type=int, default=1, help="phrasings per clip (default: 1)")
    data_p.add_argument("--caption-dropout", type=float, default=0.1, help="share of rows with no caption (default: 0.1)")
    data_p.add_argument("--min-sec", type=float, default=3.0)
    data_p.add_argument("--max-sec", type=float, default=20.0)
    data_p.add_argument("--max-scan-rows", type=int, default=60_000, help="audio-pass safety limit")
    data_p.add_argument("--force", action="store_true", help="rebuild even if the config matches")

    model_p = add("model", "download openbmb/VoxCPM2 into a local directory")
    model_p.add_argument("--revision", default=None)
    model_p.add_argument("--force", action="store_true", help="re-download even if the snapshot looks valid")

    train_p = add("train", "caption-conditioned LoRA fine-tuning (resumes automatically)")
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
    train_p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="extra YAML key (repeatable)"
    )

    eval_p = add("eval", "design prompts + caption-adherence metrics per checkpoint")
    eval_p.add_argument("--steps", type=int, nargs="*", default=None, help="checkpoint step(s); default: latest")
    eval_p.add_argument("--ab", action="store_true", help="also synthesize with LoRA disabled")
    eval_p.add_argument("--limit", type=int, default=None, help="only the first N design prompts")
    eval_p.add_argument("--optimize", action="store_true", help="torch.compile (slower start, faster clips)")
    eval_p.add_argument("--device", default="cuda")

    export_p = add("export", "package the LoRA adapter + eval samples + design palette")
    export_p.add_argument("--step", type=int, default=None, help="default: latest checkpoint")
    export_p.add_argument("--no-samples", action="store_true", help="do not copy eval wavs into the package")

    design_p = add("design", "synthesize one clip from a voice description")
    design_p.add_argument("--caption", required=True, help='e.g. "A Kenyan female voice in her thirties, mid-pitched"')
    design_p.add_argument("--text", required=True, help="what the voice should say")
    design_p.add_argument("--out", default=None, help="output wav (default: exports/<run>/design/*.wav)")
    design_p.add_argument("--lora", default=None, help="exported adapter dir (default: base model only)")
    design_p.add_argument("--seed", type=int, default=42)
    design_p.add_argument("--cfg-value", type=float, default=2.0)
    design_p.add_argument("--timesteps", type=int, default=10)

    status_p = add("status", "print what is done and what to run next")
    status_p.add_argument("--json", action="store_true", help="machine-readable output")

    sub.add_parser("doctor", help="diagnose GPU, Drive, deps, base model and data", parents=[common])
    sub.add_parser("datasets", help="list dataset presets", parents=[common])

    all_p = add("all", "run every stage, skipping the finished ones")
    all_p.add_argument("--profile", choices=sorted(PROFILES))
    all_p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    all_p.add_argument("--max-hours", type=float, default=None)
    all_p.add_argument("--allow-small-gpu", action="store_true")
    all_p.add_argument("--preset", default=DEFAULT_DATASETS[0], choices=sorted(DATASET_PRESETS))
    all_p.add_argument("--speakers", type=int, default=120)
    all_p.add_argument("--clips-per-speaker", type=int, default=30)
    all_p.add_argument("--hours", type=float, default=10.0)
    all_p.add_argument("--val-speakers", type=float, default=0.1)
    all_p.add_argument("--caption-variants", type=int, default=1)
    all_p.add_argument("--caption-dropout", type=float, default=0.1)
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


def _data_config(args: argparse.Namespace) -> D.PrepareConfig:
    return D.PrepareConfig(
        preset=getattr(args, "preset", DEFAULT_DATASETS[0]),
        dataset_id=getattr(args, "dataset", "") or "",
        split=getattr(args, "split", "") or "",
        max_speakers=args.speakers,
        max_clips_per_speaker=args.clips_per_speaker,
        min_clips_per_speaker=getattr(args, "min_clips_per_speaker", 6),
        max_audio_hours=args.hours,
        min_dur=getattr(args, "min_sec", 3.0),
        max_dur=getattr(args, "max_sec", 20.0),
        val_speaker_fraction=args.val_speakers,
        val_clips_per_speaker=getattr(args, "val_clips_per_speaker", 3),
        caption_variants=args.caption_variants,
        caption_dropout=args.caption_dropout,
        max_scan_rows=getattr(args, "max_scan_rows", 60_000),
        force=getattr(args, "force", False),
    )


def cmd_data(paths: Paths, args: argparse.Namespace) -> int:
    progress = Progress(paths)
    stats = D.prepare(paths, _data_config(args), log=log)
    progress.event(
        "data",
        "done",
        note=f"{stats['train']['clips']} train / {stats['val']['clips']} val rows, "
        f"{stats['train']['duration_hours']} h, {stats['speakers']} speakers",
        dataset=stats.get("dataset"),
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
        data=_data_config(args) if hasattr(args, "speakers") else None,
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
        log(f"[train] next: kenya-voices eval --run {args.run}")
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
    score = summary["adherence"]["score"]
    log(f"[eval] caption adherence: {score}")
    log(f"[eval] next: kenya-voices export --run {args.run}")
    return 0


def cmd_export(paths: Paths, args: argparse.Namespace) -> int:
    dest = I.export(
        paths,
        I.ExportOptions(run=args.run, step=args.step, include_samples=not args.no_samples),
        log=log,
    )
    log(f"[export] adapter package: {dest}")
    return 0


def cmd_design(paths: Paths, args: argparse.Namespace) -> int:
    out = I.design(
        paths,
        I.DesignOptions(
            caption=args.caption,
            text=args.text,
            out=Path(args.out) if args.out else None,
            lora=Path(args.lora) if args.lora else None,
            run=args.run,
            seed=args.seed,
            cfg_value=args.cfg_value,
            timesteps=args.timesteps,
        ),
        log=log,
    )
    log(f"[design] wrote {out}")
    return 0


def cmd_datasets(paths: Paths, args: argparse.Namespace) -> int:
    for name, preset in DATASET_PRESETS.items():
        default = " (default)" if name == DEFAULT_DATASETS[0] else ""
        log(f"{name}{default}")
        for key, value in preset.items():
            log(f"  {key:<10} {value}")
        log("")
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
        checks.append(("native bf16 (cc >= 8.0)", gpu.bf16, f"cc {gpu.cc:.1f} -> profile {gpu.profile}"))
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
        checks.append(("VoxCPM installed", False, "run `kenya-voices env`"))

    missing = E.missing_imports()
    checks.append(("required imports", not missing, ", ".join(missing) or "torch, torchaudio, datasets, ..."))

    checks.append(("VoxCPM repo", paths.train_script.exists(), f"{paths.voxcpm_repo} @ {E.git_rev(paths.voxcpm_repo)}"))
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
            speakers = stats.get("speakers", {})
            checks.append(
                (
                    "dataset stats",
                    True,
                    f"{stats['train'].get('duration_hours')} h, {speakers.get('train')} train / "
                    f"{speakers.get('val')} val speakers",
                )
            )
        except (json.JSONDecodeError, KeyError):
            checks.append(("dataset stats", False, str(paths.data_stats)))

    log(f"{BASE_MODEL_ID} voice design on {DEFAULT_DATASETS[0]} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    for label, ok, detail in checks:
        log(f"[{'ok  ' if ok else 'FAIL'}] {label:<28} {detail}")
    log("")
    if on_colab:
        log(f"Colab detected. Recommended profile: {gpu.profile}")
    failures = [label for label, ok, _ in checks if not ok]
    if failures:
        log(f"{len(failures)} check(s) failing: {', '.join(failures)}")
        log("Fix with `kenya-voices env`, `kenya-voices data`, `kenya-voices model` as appropriate.")
        return 1
    log("All checks passed - `kenya-voices all --resume` is safe to run.")
    return 0


def _setup_ready(paths: Paths) -> bool:
    """Is the *VM-local* install still valid? (deps live in the VM, not on Drive)"""
    return E.marker_path().exists() and paths.train_script.exists() and not E.missing_imports()


def _stage_already_done(paths: Paths, args: argparse.Namespace, stage: str, progress: Progress) -> bool:
    if args.force:
        return False
    if stage == "setup":
        return _setup_ready(paths)
    if stage == "train":
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
        preset=args.preset,
        dataset="",
        split="",
        speakers=args.speakers,
        clips_per_speaker=args.clips_per_speaker,
        min_clips_per_speaker=6,
        hours=args.hours,
        val_speakers=args.val_speakers,
        val_clips_per_speaker=3,
        caption_variants=args.caption_variants,
        caption_dropout=args.caption_dropout,
        min_sec=3.0,
        max_sec=20.0,
        max_scan_rows=60_000,
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
        speakers=args.speakers,
        clips_per_speaker=args.clips_per_speaker,
        min_clips_per_speaker=6,
        hours=args.hours,
        val_speakers=args.val_speakers,
        val_clips_per_speaker=3,
        caption_variants=args.caption_variants,
        caption_dropout=args.caption_dropout,
        min_sec=3.0,
        max_sec=20.0,
        preset=args.preset,
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
    "design": cmd_design,
    "status": cmd_status,
    "doctor": cmd_doctor,
    "datasets": cmd_datasets,
    "all": cmd_all,
}


def _ensure_artifacts_root(args: argparse.Namespace) -> None:
    """On Colab, mount Drive *before* resolving paths (else checkpoints are lost)."""
    if args.command == "doctor" or args.artifacts or os.environ.get("KENYA_ARTIFACTS"):
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