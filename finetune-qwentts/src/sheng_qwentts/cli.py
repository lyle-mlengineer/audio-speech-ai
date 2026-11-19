"""``sheng-qwentts`` - the Colab-first driver for the Qwen3-TTS Sheng fine-tune.

Every stage is also usable on its own (``sheng-qwentts data``, ``... codes``,
``... train``) and every one of them is idempotent, so after a disconnect you run
``sheng-qwentts all --resume`` and it picks up where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import data as data_stage
from .config import PROFILES, SMALL_GPU, TrainConfig, allowed_profiles
from .paths import Paths, latest_checkpoint, resolve_artifacts
from .progress import Progress


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sheng-qwentts", description=__doc__)
    p.add_argument("--artifacts", default=None, help="artifacts root (default: Drive on Colab)")
    p.add_argument("--run", default=None, help="run name (default: sheng_06b)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="what is done, what is left")
    sub.add_parser("setup", help="check environment / GPU and print the install hint")

    d = sub.add_parser("data", help="stage 2: dataset -> 24 kHz WAVs + manifests")
    d.add_argument("--val-size", type=int, default=4)
    d.add_argument("--ref-index", type=int, default=None)
    d.add_argument("--force", action="store_true")

    m = sub.add_parser("model", help="stage 3: download the Base model + 12 Hz tokenizer")
    m.add_argument("--force", action="store_true")

    c = sub.add_parser("codes", help="stage 4: extract audio_codes (needs GPU)")
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--force", action="store_true")

    t = sub.add_parser("train", help="stage 5: SFT")
    _add_train_args(t)

    e = sub.add_parser("eval", help="stage 6: synthesise with the checkpoint")
    e.add_argument("--checkpoint", default=None, help="checkpoint dir (default: latest)")
    e.add_argument("--compare-base", action="store_true", help="also zero-shot clone with the Base model")
    e.add_argument("--dtype", default="bfloat16")
    e.add_argument("--attn", default="sdpa")
    e.add_argument("--device", default="cuda:0")

    x = sub.add_parser("export", help="stage 7: package model + samples + model card")
    x.add_argument("--checkpoint", default=None)
    x.add_argument("--zip", action="store_true")

    a = sub.add_parser("all", help="setup -> data -> model -> codes -> train -> eval -> export")
    _add_train_args(a)
    a.add_argument("--skip-eval", action="store_true")
    a.add_argument("--val-size", type=int, default=4)
    a.add_argument("--compare-base", action="store_true")
    return p


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default=None, choices=sorted(PROFILES))
    p.add_argument("--allow-small-gpu", action="store_true", help="allow t4/cpu profiles")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--save-interval", type=int, default=None)
    p.add_argument("--keep-last", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true", help="2 steps - verifies the pipeline only")
    p.add_argument("--no-resume", action="store_true",
                   help="start training over and redo finished stages")


# ----------------------------------------------------------------------- utils
def _paths(args) -> Paths:
    paths = Paths(resolve_artifacts(args.artifacts))
    paths.ensure()
    paths.progress_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _run_name(args) -> str:
    from .paths import DEFAULT_RUN

    return args.run or DEFAULT_RUN


def _train_config(args, paths: Paths) -> TrainConfig:
    from .env import report, suggest_profile

    profile = args.profile
    if profile is None:
        profile = suggest_profile(report())
        print(f"[config] no --profile given, detected {profile!r}")
    if profile in SMALL_GPU and not args.allow_small_gpu:
        raise SystemExit(
            f"profile {profile!r} is a smoke-test profile - pass --allow-small-gpu if that is really what you want"
        )
    cfg = TrainConfig.from_profile(
        profile,
        run=_run_name(args),
        steps=args.steps,
        learning_rate=args.lr,
        save_interval=args.save_interval,
        keep_last=args.keep_last,
    )
    cfg.write_yaml(paths)
    return cfg


def _python(argv: list[str], *, src_dir: Path) -> int:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) + (os.pathsep + existing if existing else "")
    print("[run]", " ".join([sys.executable, "-m", *argv[:2], "..."]), flush=True)
    return subprocess.call([sys.executable, "-m", *argv], env=env)


# --------------------------------------------------------------------- stages
def cmd_setup(args) -> int:
    from .env import run as env_run

    info = env_run()
    prog = Progress(_paths(args), _run_name(args))
    # only a missing torch/qwen-tts is fatal; a GPU-less box can still do
    # data/model/export, and transformers drift is a warning.
    fatal = info["torch"] is None or info["qwen_tts"] is None
    prog.mark("setup", "failed" if fatal else "done",
              f"{info['gpu']} / {info['suggested_profile']}"
              + ("" if not info["problems"] else f" / {len(info['problems'])} warning(s)"))
    return 1 if fatal else 0


def cmd_data(args) -> int:
    paths = _paths(args)
    prog = Progress(paths, _run_name(args))
    prog.mark("data", "running")
    info = data_stage.build(paths, val_size=args.val_size, ref_index=args.ref_index, force=args.force)
    detail = (
        f"{info['clips']} clips / {info['total_minutes']:.1f} min @ {info['sample_rate']} Hz "
        f"-> {info['train_clips']} train + {info['val_clips']} val"
    )
    prog.mark("data", "done", detail)
    print(f"[data] {detail}")
    print(f"[data] report: {paths.data_report}")
    return 0


def cmd_model(args) -> int:
    paths = _paths(args)
    prog = Progress(paths, _run_name(args))
    prog.mark("model", "running")
    from .model import download_snapshots

    got = download_snapshots(paths, force=args.force)
    size_gb = sum(f.stat().st_size for f in paths.model_dir.rglob("*") if f.is_file()) / 1024**3
    prog.mark("model", "done", f"0.6B Base {size_gb:.2f} GB + tokenizer")
    for repo, dest in got.items():
        print(f"[model] {repo} -> {dest}")
    return 0


def cmd_codes(args) -> int:
    paths = _paths(args)
    prog = Progress(paths, _run_name(args))
    prog.mark("codes", "running")
    from . import codes as codes_stage

    out = codes_stage.run(paths, device=args.device, batch=args.batch, force=args.force)
    detail = f"{out['train']} train / {out.get('val', 0)} val clips, {out['frames_mean']} frames avg"
    prog.mark("codes", "done", detail)
    print(f"[codes] {detail}")
    return 0


def cmd_train(args) -> int:
    paths = _paths(args)
    run = _run_name(args)
    prog = Progress(paths, run)
    cfg = _train_config(args, paths)
    if not paths.train_codes_jsonl.is_file():
        raise SystemExit("missing train_with_codes.jsonl - run `sheng-qwentts codes` first")
    argv = cfg.trainer_args(paths)
    argv += ["--val-jsonl", str(paths.val_codes_jsonl)]
    if args.max_hours:
        argv += ["--max-hours", str(args.max_hours)]
    if args.smoke:
        argv += ["--smoke"]
    if args.no_resume:
        argv += ["--no-resume"]
    prog.mark("train", "running", f"{cfg.steps} steps @ {cfg.profile}")
    prog.note("train", cfg.to_dict())
    code = _python(["sheng_qwentts.train", *argv], src_dir=Path(__file__).resolve().parents[1])
    ckpt = latest_checkpoint(paths, run)
    if code == 0 and ckpt is not None:
        prog.mark("train", "done", f"{ckpt.name} ({cfg.steps} steps)")
    else:
        prog.mark("train", "failed", f"exit {code}")
    print(f"[train] checkpoint: {ckpt}")
    return code


def cmd_eval(args) -> int:
    paths = _paths(args)
    run = _run_name(args)
    prog = Progress(paths, run)
    from . import infer

    prog.mark("eval", "running")
    report = infer.run_eval(
        paths, run,
        checkpoint=getattr(args, "checkpoint", None),
        dtype=getattr(args, "dtype", "bfloat16"),
        attn=getattr(args, "attn", "sdpa"),
        device=getattr(args, "device", "cuda:0"),
        compare_base=getattr(args, "compare_base", False),
    )
    bad = report["problems"]
    prog.mark("eval", "failed" if bad else "done",
              f"{len(report['finetuned'])} samples" + (f", {len(bad)} suspicious" if bad else ""))
    print(f"[eval] {paths.eval_dir(run)}")
    if bad:
        print("[eval] suspicious output - listen before exporting:", ", ".join(Path(b).name for b in bad))
    return 0


def cmd_export(args) -> int:
    paths = _paths(args)
    run = _run_name(args)
    prog = Progress(paths, run)
    from . import infer

    prog.mark("export", "running")
    dest = infer.export(paths, run, checkpoint=args.checkpoint, zip_bundle=args.zip)
    prog.mark("export", "done", str(dest))
    print(f"[export] {dest}")
    return 0


def cmd_status(args) -> int:
    paths = _paths(args)
    prog = Progress(paths, _run_name(args))
    print(prog.render_console())
    ckpt = latest_checkpoint(paths, _run_name(args))
    if ckpt:
        print(f"\nlatest checkpoint: {ckpt}")
        state_file = ckpt / "train_state.json"
        if state_file.is_file():
            print("   " + json.dumps(json.loads(state_file.read_text(encoding="utf-8")))[:300])
    return 0


def cmd_all(args) -> int:
    paths = _paths(args)
    run = _run_name(args)
    prog = Progress(paths, run)
    skip_done = not args.no_resume
    force = not skip_done

    steps = [
        ("setup", lambda: cmd_setup(args)),
        ("data", lambda: cmd_data(argparse.Namespace(**{**vars(args), "val_size": args.val_size,
                                                       "ref_index": None, "force": False}))),
        ("model", lambda: cmd_model(argparse.Namespace(**{**vars(args), "force": False}))),
        ("codes", lambda: cmd_codes(argparse.Namespace(**{**vars(args), "batch": 8,
                                                         "device": "cuda:0", "force": False}))),
        ("train", lambda: cmd_train(args)),
        ("eval", lambda: None if args.skip_eval else cmd_eval(args)),
        ("export", lambda: cmd_export(argparse.Namespace(**{**vars(args), "checkpoint": None, "zip": False}))),
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
    print("\n[all] finished; check `sheng-qwentts status`")
    return 0


# ----------------------------------------------------------------------- entry
COMMANDS = {
    "status": cmd_status,
    "setup": cmd_setup,
    "data": cmd_data,
    "model": cmd_model,
    "codes": cmd_codes,
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
