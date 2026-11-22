"""``sauti-sheng`` - the Colab-first driver for the Sauti TTS Sheng fine-tune.

Every stage is also usable on its own and every one is idempotent, so after a
Colab disconnect you run ``sauti-sheng all --resume`` and it picks up where it
stopped:

.. code-block:: text

    setup -> data -> model -> train -> eval -> export
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import data as data_stage
from .config import DEFAULT_PROFILE, PROFILES, resolve_profile, write_config
from .paths import DEFAULT_RUN, Paths, resolve_artifacts, resolve_scratch
from .progress import Progress


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sauti-sheng", description=__doc__)
    p.add_argument("--artifacts", default=None, help="artifacts root (default: Drive on Colab)")
    p.add_argument("--scratch", default=None, help="fast local disk for the 5.4 GB raw checkpoint")
    p.add_argument("--run", default=DEFAULT_RUN, help=f"run name (default: {DEFAULT_RUN})")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="what is done, what is left")
    sub.add_parser("setup", help="check environment / GPU and print the install hint")

    d = sub.add_parser("data", help="stage 2: parquet -> 24 kHz WAVs + F5 dataset + reference clip")
    d.add_argument("--val-size", type=int, default=4)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--ref-index", type=int, default=None, help="force a validation clip for the reference")
    d.add_argument("--ref-seconds", type=float, default=9.0)
    d.add_argument("--force", action="store_true")

    m = sub.add_parser("model", help="stage 3: extract the EMA weights from the 5.4 GB release")
    m.add_argument("--force", action="store_true")

    t = sub.add_parser("train", help="stage 4: F5-TTS fine-tune")
    _add_train_args(t)

    e = sub.add_parser("eval", help="stage 5: synthesise Sheng probes with the checkpoint")
    e.add_argument("--checkpoint", default=None, help="default: <run>/model_last.pt")
    e.add_argument("--use-ema", action="store_true",
                   help="load EMA weights (default: online weights, which are better early in a fine-tune)")
    e.add_argument("--compare-base", action="store_true", help="also run the untouched Sauti model")
    e.add_argument("--nfe-step", type=int, default=32)
    e.add_argument("--device", default="cuda")

    x = sub.add_parser("export", help="stage 6: weights + vocab + reference + samples")
    x.add_argument("--checkpoint", default=None)
    x.add_argument("--use-ema", action="store_true")
    x.add_argument("--zip", action="store_true")

    a = sub.add_parser("all", help="setup -> data -> model -> train -> eval -> export")
    _add_train_args(a)
    a.add_argument("--skip-eval", action="store_true")
    a.add_argument("--compare-base", action="store_true")
    a.add_argument("--val-size", type=int, default=4)
    a.add_argument("--no-resume", action="store_true", help="redo finished stages (training still resumes)")
    return p


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default=None, choices=sorted(PROFILES),
                   help=f"force a GPU profile (default: auto-detect; {DEFAULT_PROFILE} is the default runtime)")
    p.add_argument("--allow-small-gpu", action="store_true",
                   help="deprecated, no effect: the 16 GB t4 profile is the default")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--save-interval", type=int, default=None)
    p.add_argument("--last-interval", type=int, default=None)
    p.add_argument("--keep-last", type=int, default=None)
    p.add_argument("--mixed-precision", default=None, choices=["bf16", "fp16", "no"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fresh", action="store_true", help="delete model_*.pt and restart from the pretrained init")
    p.add_argument("--smoke", action="store_true", help="1 epoch x 4 clips: verifies wiring only")


# ----------------------------------------------------------------------- utils
def _paths(args) -> Paths:
    paths = Paths(resolve_artifacts(args.artifacts), scratch=resolve_scratch(args.scratch))
    paths.ensure()
    return paths


def _run_name(args) -> str:
    return args.run or DEFAULT_RUN


def _profile(args):
    from .env import report

    info = report()
    try:
        profile = resolve_profile(args.profile, info.get("gpu"), info.get("gpu_gb"), args.allow_small_gpu)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"[config] profile {profile.name!r}: {profile.note}")
    return profile


def _python(module: str, argv: list[str]) -> int:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"[run] {sys.executable} -m {module} {' '.join(argv)}", flush=True)
    return subprocess.call([sys.executable, "-m", module, *argv], env=env)


# --------------------------------------------------------------------- stages
def cmd_setup(args) -> int:
    from .env import run as env_run

    paths = _paths(args)
    info = env_run(paths.artifacts, paths.scratch)
    prog = Progress(paths, _run_name(args))
    fatal = info["torch"] is None or info["f5_tts"] is None
    prog.mark(
        "setup",
        "failed" if fatal else "done",
        f"{info['gpu']} / {info['suggested_profile']}"
        + ("" if not info["problems"] else f" / {len(info['problems'])} warning(s)"),
    )
    return 1 if fatal else 0


def cmd_data(args) -> int:
    paths = _paths(args)
    prog = Progress(paths, _run_name(args))
    prog.mark("data", "running")
    stats = data_stage.prepare(
        paths,
        val_size=getattr(args, "val_size", 4),
        seed=getattr(args, "seed", 42),
        ref_index=getattr(args, "ref_index", None),
        ref_seconds=getattr(args, "ref_seconds", 9.0),
        force=getattr(args, "force", False),
    )
    detail = (
        f"{stats['n_clips']} clips / {stats['total_seconds'] / 60:.1f} min @ {stats['sample_rate']} Hz "
        f"-> {stats['n_train']} train + {stats['n_val']} val"
    )
    prog.mark("data", "done", detail)
    print(f"[data] {detail}")
    print(f"[data] report: {paths.data_report}")
    return 0


def cmd_model(args) -> int:
    paths = _paths(args)
    prog = Progress(paths, _run_name(args))
    prog.mark("model", "running")
    from .model import extract

    summary = extract(paths, force=getattr(args, "force", False))
    detail = (
        f"EMA {summary['ema_bytes'] / 1024**3:.2f} GB"
        + (f" (raw {summary['raw_bytes'] / 1024**3:.2f} GB deleted)" if summary.get("raw_deleted") else "")
    )
    prog.mark("model", "done", detail)
    print(f"[model] {paths.sauti_ema_ckpt}  {detail}")
    print(f"[model] report: {paths.model_report}")
    return 0


def cmd_train(args) -> int:
    paths = _paths(args)
    run = _run_name(args)
    prog = Progress(paths, run)
    profile = _profile(args)

    batch = args.batch_size or profile.batch_size
    accum = args.grad_accum or profile.grad_accum
    epochs = args.epochs or profile.epochs
    lr = args.lr or profile.learning_rate
    mixed = args.mixed_precision or profile.mixed_precision
    save_every = args.save_interval or profile.save_per_updates
    last_every = args.last_interval or profile.last_per_updates
    keep = args.keep_last if args.keep_last is not None else profile.keep_last

    write_config(
        paths, profile, run,
        epochs=epochs, batch_size=batch, grad_accum=accum, learning_rate=lr,
        mixed_precision=mixed, save_per_updates=save_every, last_per_updates=last_every,
        keep_last=keep, max_hours=args.max_hours or None,
    )

    argv = [
        "--artifacts", str(paths.artifacts),
        "--scratch", str(paths.scratch or ""),
        "--run", run,
        "--epochs", str(epochs),
        "--batch-size", str(batch),
        "--grad-accum", str(accum),
        "--lr", str(lr),
        "--warmup", str(profile.warmup_updates),
        "--save-per-updates", str(save_every),
        "--last-per-updates", str(last_every),
        "--keep-last", str(keep),
        "--mixed-precision", mixed,
        "--num-workers", str(args.num_workers),
    ]
    if args.max_hours:
        argv += ["--max-hours", str(args.max_hours)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.fresh:
        argv += ["--fresh"]
    if args.smoke:
        argv += ["--smoke"]

    prog.mark("train", "running", f"{epochs} epochs, batch {batch}x{accum} @ {profile.name}")
    code = _python("sauti_sheng.train", argv)
    ckpt = paths.latest_checkpoint(run)
    if code == 0 and ckpt.is_file():
        prog.mark("train", "done", f"{ckpt.name} ({ckpt.stat().st_size / 1024**3:.2f} GB)")
    else:
        prog.mark("train", "failed", f"exit {code}")
    print(f"[train] checkpoint: {ckpt}")
    return code


def cmd_eval(args) -> int:
    paths = _paths(args)
    run = _run_name(args)
    prog = Progress(paths, run)
    from .eval import run_eval

    prog.mark("eval", "running")
    report = run_eval(
        paths, run,
        checkpoint=getattr(args, "checkpoint", None),
        use_ema=getattr(args, "use_ema", False),
        compare_base=getattr(args, "compare_base", False),
        device=getattr(args, "device", "cuda"),
        nfe_step=getattr(args, "nfe_step", 32),
    )
    bad = report["problems"]
    prog.mark("eval", "failed" if bad else "done",
              f"{len(report['finetuned'])} samples" + (f", {len(bad)} suspicious" if bad else ""))
    print(f"[eval] {report['dir']}")
    if bad:
        print("[eval] suspicious output - listen before exporting:", ", ".join(Path(b).name for b in bad))
    return 0


def cmd_export(args) -> int:
    paths = _paths(args)
    run = _run_name(args)
    prog = Progress(paths, run)
    from .export import export

    prog.mark("export", "running")
    dest = export(
        paths, run,
        checkpoint=getattr(args, "checkpoint", None),
        use_ema=getattr(args, "use_ema", False),
        zip_bundle=getattr(args, "zip", False),
    )
    prog.mark("export", "done", str(dest))
    print(f"[export] {dest}")
    return 0


def cmd_status(args) -> int:
    paths = _paths(args)
    run = _run_name(args)
    prog = Progress(paths, run)
    print(prog.render_console())
    ckpt = paths.latest_checkpoint(run)
    if ckpt.is_file():
        print(f"\nlatest checkpoint: {ckpt} ({ckpt.stat().st_size / 1024**3:.2f} GB)")
    if paths.sauti_ema_ckpt.is_file():
        print(f"sauti EMA:         {paths.sauti_ema_ckpt}")
    return 0


def cmd_all(args) -> int:
    paths = _paths(args)
    run = _run_name(args)
    prog = Progress(paths, run)
    skip_done = not args.no_resume

    steps = [
        ("setup", lambda: cmd_setup(args)),
        ("data", lambda: cmd_data(argparse.Namespace(
            artifacts=args.artifacts, scratch=args.scratch, run=run,
            val_size=args.val_size, seed=42, ref_index=None, ref_seconds=9.0, force=False))),
        ("model", lambda: cmd_model(argparse.Namespace(
            artifacts=args.artifacts, scratch=args.scratch, run=run, force=False))),
        ("train", lambda: cmd_train(args)),
        ("eval", lambda: None if args.skip_eval else cmd_eval(argparse.Namespace(
            **{**vars(args), "checkpoint": None, "use_ema": False, "device": "cuda", "nfe_step": 32}))),
        ("export", lambda: cmd_export(argparse.Namespace(
            artifacts=args.artifacts, scratch=args.scratch, run=run,
            checkpoint=None, use_ema=False, zip=False))),
    ]
    for stage, fn in steps:
        if skip_done and prog.is_done(stage):
            print(f"[all] {stage}: already done, skipping (use --no-resume to redo)")
            continue
        print(f"\n[all] === {stage} ===", flush=True)
        code = fn()
        if code not in (0, None):
            prog.mark(stage, "failed", f"exit {code}")
            return code
    print("\n[all] finished; check `sauti-sheng status`")
    return 0


# ----------------------------------------------------------------------- entry
COMMANDS = {
    "status": cmd_status,
    "setup": cmd_setup,
    "data": cmd_data,
    "model": cmd_model,
    "train": cmd_train,
    "eval": cmd_eval,
    "export": cmd_export,
    "all": cmd_all,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
