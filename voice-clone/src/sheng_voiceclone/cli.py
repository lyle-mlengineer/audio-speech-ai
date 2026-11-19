"""``sheng-voiceclone`` - the driver.

    status     where am I, what is next
    setup      environment + GPU -> profile
    data       dataset -> 16 kHz WAVs, splits, reference clip
    model      download the Chatterbox Multilingual snapshot
    baseline   zero-shot clone of the reference voice, synthesised and scored
    train      fine-tune T3 (delegated to finetune-chatterbox)
    eval       score a fine-tuned checkpoint with the same prompts
    score      score/re-score any existing eval directory
    compare    baseline vs tuned, with a verdict
    all        everything above, resumable

Every stage is idempotent and records itself in ``progress.md``, so a dead Colab
VM costs you a re-run, not the run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import config as cfg
from . import dataset, evals, progress
from .paths import (
    DEFAULT_RUN,
    MODEL_REPO_ID,
    Paths,
    default_root,
    resolve,
)

class CliError(RuntimeError):
    """A stage cannot run; the message is the instruction, so print it plainly."""


PRETRAINED_PATTERNS = [
    "ve.pt",
    "t3_*.safetensors",
    "s3gen.pt",
    "conds.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "tokenizer_config.json",
]


# ------------------------------------------------------------------- helpers
def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        pass
    return 0.0


def _say(msg: str) -> None:
    print(msg, flush=True)


def _common(p: argparse.ArgumentParser, *, artifacts: bool = True) -> None:
    if artifacts:
        p.add_argument("--artifacts", default=None, help=f"artifacts root (default: {default_root()})")
    p.add_argument("--run", default=None, help="run name (default: sheng_clone / from state)")


def _paths(args: argparse.Namespace) -> Paths:
    paths = resolve(getattr(args, "artifacts", None))
    paths.ensure_dirs()
    return paths


def _run_name(paths: Paths, args: argparse.Namespace) -> str:
    if getattr(args, "run", None):
        return str(args.run)
    state = progress.load_state(paths)
    return str(state.get("run") or DEFAULT_RUN)


def _judges(args: argparse.Namespace) -> tuple[object | None, object | None]:
    device = getattr(args, "device", None) or _device()
    transcribe = None
    embed = None
    if getattr(args, "asr", False):
        _say(f"  loading ASR judge ({args.asr_model}) on {device} ...")
        transcribe = evals.asr_judge(args.asr_model, device=device)
    if getattr(args, "sim", False):
        _say(f"  loading speaker encoder ({args.spk_model}) on {device} ...")
        embed = evals.spk_judge(args.spk_model, device=device)
    return transcribe, embed


def _require_ref(paths: Paths) -> Path:
    if not paths.ref_audio.is_file():
        raise CliError(
            f"no reference clip at {paths.ref_audio} - run `sheng-voiceclone data` first"
        )
    return paths.ref_audio


# -------------------------------------------------------------------- stages
def cmd_status(args: argparse.Namespace) -> int:
    paths = _paths(args)
    _say(progress.summary(paths))
    if paths.data_dir.joinpath("data_info.json").is_file():
        info = json.loads((paths.data_dir / "data_info.json").read_text(encoding="utf-8"))
        _say(f"\ndata: {info['train']} train / {info['val']} val / {info['test']} test, "
             f"{info['minutes']} min, speaker {info['speaker']}")
    for name in ("baseline", "tuned"):
        p = paths.scores / f"{name}.json"
        if p.is_file():
            s = json.loads(p.read_text(encoding="utf-8"))["summary"]
            _say(f"{name:<9} cer={s['cer_mean']} wer={s['wer_mean']} sim={s['similarity_mean']} "
                 f"ok={s['ok']}/{s['clips']}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    paths = _paths(args)
    ok = True
    _say(f"python      {sys.version.split()[0]}")
    for mod in ("numpy", "soundfile", "pyarrow", "huggingface_hub"):
        try:
            m = __import__(mod)
            _say(f"{mod:<11} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:
            ok = False
            _say(f"{mod:<11} MISSING ({exc})")
    try:
        import torch

        _say(f"torch        {torch.__version__} (cuda {torch.version.cuda}, available {torch.cuda.is_available()})")
        vram = _vram_gb()
        if vram:
            _say(f"gpu          {torch.cuda.get_device_name(0)} ({vram:.0f} GB)")
    except Exception as exc:
        _say(f"torch        MISSING ({exc}) - synthesis needs torch; data/score do not")
        vram = 0.0
    try:
        import chatterbox  # noqa: F401

        _say("chatterbox   installed")
    except Exception:
        _say("chatterbox   not installed -> `pip install chatterbox-tts` for synth")

    prof = args.profile if args.profile != "auto" else cfg.pick_profile(vram)
    yaml_path = cfg.write_yaml(cfg.profile(prof), paths, run=_run_name(paths, args))
    _say(f"profile      {prof} (effectively batch {cfg.profile(prof).effective_batch}) -> {yaml_path}")
    if vram and vram < cfg.profile(prof).min_vram_gb and not cfg.profile(prof).allow_small_gpu:
        _say(f"WARNING: {vram:.0f} GB VRAM < {cfg.profile(prof).min_vram_gb} GB for profile '{prof}'")
        ok = False
    progress.record(paths, "setup", "done" if ok else "failed", detail=f"profile {prof}", run=_run_name(paths, args))
    return 0 if ok else 1


def cmd_data(args: argparse.Namespace) -> int:
    paths = _paths(args)
    info = dataset.build(
        paths,
        source=args.source,
        val_clips=args.val_clips,
        test_clips=args.test_clips,
        seed=args.seed,
        force=args.force,
    )
    _say(
        f"data: {info['clips']} clips / {info['minutes']} min @ {info['sample_rate']} Hz -> "
        f"{info['train']} train / {info['val']} val / {info['test']} test"
    )
    _say(f"      reference: {info['ref_audio']} ({info['ref_seconds']} s)")
    progress.record(
        paths,
        "data",
        "done",
        detail=f"{info['clips']} clips / {info['minutes']} min -> {info['train']}/{info['val']}/{info['test']}",
        run=_run_name(paths, args),
    )
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    paths = _paths(args)
    from huggingface_hub import snapshot_download

    _say(f"model: {MODEL_REPO_ID} -> {paths.pretrained_dir}")
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=str(paths.pretrained_dir),
        allow_patterns=PRETRAINED_PATTERNS,
    )
    files = sorted(p.name for p in paths.pretrained_dir.iterdir() if p.is_file())
    _say(f"      {len(files)} files: {', '.join(files[:8])}{' ...' if len(files) > 8 else ''}")
    progress.record(paths, "model", "done", detail=f"{len(files)} files", run=_run_name(paths, args))
    return 0


def _synth_and_score(
    paths: Paths,
    args: argparse.Namespace,
    *,
    name: str,
    title: str,
    tuned_t3: Path | None,
    stage: str,
) -> dict[str, object]:
    ref = _require_ref(paths)
    out_dir = paths.evals / name
    rows = evals.plan(evals.eval_prompts(args.prompts), out_dir, tag=name)
    evals.write_plan(rows, out_dir)
    if args.dry_run:
        _say(f"dry run: {len(rows)} prompts planned -> {out_dir / 'plan.json'}")
        return {"summary": None, "rows": rows}

    evals.synthesize(
        rows,
        ref_audio=ref,
        language=args.language,
        device=args.device or _device(),
        tuned_t3=tuned_t3,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        seed=args.seed,
        force=args.force,
    )
    transcribe, embed = _judges(args)
    scored, summary = evals.score_rows(rows, ref_audio=ref, transcribe=transcribe, embed=embed)
    js, md = evals.write_scores(paths, name, scored, summary, title=title)
    _say(
        f"scores: cer={summary['cer_mean']} wer={summary['wer_mean']} "
        f"sim={summary['similarity_mean']} ok={summary['ok']}/{summary['clips']} -> {md}"
    )
    progress.record(
        paths,
        stage,
        "done",
        detail=f"cer={summary['cer_mean']} sim={summary['similarity_mean']} ok={summary['ok']}/{summary['clips']}",
        run=_run_name(paths, args),
    )
    return {"summary": summary, "rows": scored, "json": str(js)}


def cmd_baseline(args: argparse.Namespace) -> int:
    paths = _paths(args)
    _synth_and_score(
        paths,
        args,
        name="baseline",
        title="Zero-shot clone - Chatterbox Multilingual (base T3)",
        tuned_t3=None,
        stage="baseline",
    )
    return 0


def _find_ckpt(paths: Paths, run: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():
            p = p / "t3_cfg.safetensors"
        if not p.is_file():
            raise CliError(f"checkpoint not found: {p}")
        return p
    latest = paths.checkpoints / run / "latest"
    for cand in (latest / "t3_cfg.safetensors", latest / "model.safetensors"):
        if cand.is_file():
            return cand
    dirs = sorted((paths.checkpoints / run).glob("step_*")) if (paths.checkpoints / run).is_dir() else []
    for d in reversed(dirs):
        for cand in (d / "t3_cfg.safetensors", d / "model.safetensors"):
            if cand.is_file():
                return cand
    raise CliError(
        f"no checkpoint under {paths.checkpoints / run} - run `train` first, or pass --ckpt"
    )


def cmd_eval(args: argparse.Namespace) -> int:
    paths = _paths(args)
    run = _run_name(paths, args)
    ckpt = _find_ckpt(paths, run, args.ckpt)
    _say(f"eval: {ckpt}")
    stamp = ckpt.parent.name
    name = args.name or (f"tuned_{stamp.replace('step_', '')}" if stamp.startswith("step_") else "tuned")
    _synth_and_score(
        paths,
        args,
        name=name,
        title=f"Fine-tuned clone - {ckpt}",
        tuned_t3=ckpt,
        stage="eval",
    )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Re-score an eval directory without synthesising anything."""
    paths = _paths(args)
    ref = _require_ref(paths)
    plan_path = paths.evals / args.dir / "plan.json"
    if not plan_path.is_file():
        raise CliError(f"no plan.json in {paths.evals / args.dir}")
    rows = json.loads(plan_path.read_text(encoding="utf-8"))
    transcribe, embed = _judges(args)
    scored, summary = evals.score_rows(rows, ref_audio=ref, transcribe=transcribe, embed=embed)
    _, md = evals.write_scores(paths, args.dir, scored, summary, title=f"Re-scored - {args.dir}")
    _say(
        f"scores: cer={summary['cer_mean']} sim={summary['similarity_mean']} "
        f"ok={summary['ok']}/{summary['clips']} -> {md}"
    )
    return 0


def _resolve_tuned(paths: Paths, requested: str) -> str:
    """``compare`` without a name picks the newest scored checkpoint."""
    if requested != "tuned" or (paths.scores / "tuned.json").is_file():
        return requested
    cands = sorted(p.stem for p in paths.scores.glob("tuned_*.json"))
    return cands[-1] if cands else requested


def cmd_compare(args: argparse.Namespace) -> int:
    paths = _paths(args)
    a = evals.read_scores(paths, args.baseline)
    tuned = _resolve_tuned(paths, args.tuned)
    b = evals.read_scores(paths, tuned)
    table = evals.compare(a, b)
    out = paths.scores / "compare.md"
    out.write_text(
        f"# Baseline vs tuned ({args.baseline} -> {tuned})\n\n{table}\n", encoding="utf-8"
    )
    _say(table)
    _say(f"\nwritten: {out}")
    progress.record(paths, "compare", "done", detail=tuned, run=_run_name(paths, args))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Fine-tune T3 by delegating to the finetune-chatterbox trainer."""
    paths = _paths(args)
    run = _run_name(paths, args)
    _require_ref(paths)
    if not paths.train_manifest.is_file():
        raise CliError(f"no manifest at {paths.train_manifest} - run `sheng-voiceclone data`")
    prof = cfg.override(
        cfg.profile(args.profile) if args.profile != "auto" else cfg.profile(cfg.pick_profile(_vram_gb())),
        steps=args.steps,
        lr=args.lr,
        lora=args.lora,
    )
    yaml_path = cfg.write_yaml(prof, paths, run=run)
    _say(f"train: profile {prof.name}, {prof.steps} steps, lr {prof.lr}, lora={prof.lora}")
    _say(f"       config -> {yaml_path}")

    try:
        import importlib.util

        has = importlib.util.find_spec("sheng_chatterbox") is not None
    except Exception:
        has = False
    if not has:
        _say("trainer not installed: pip install -e ../finetune-chatterbox (same Colab VM)")
        _say(f"then: sheng-chatterbox train --run {run} --profile {prof.name} "
             f"--steps {prof.steps} --lr {prof.lr}" + (" --lora" if prof.lora else ""))
        return 1
    if args.dry_run:
        _say(f"dry run: would run python -m sheng_chatterbox train --run {run} --profile {prof.name}")
        return 0

    argv = [
        sys.executable,
        "-m",
        "sheng_chatterbox.cli",
        "train",
        "--run",
        run,
        "--profile",
        prof.name,
        "--steps",
        str(prof.steps),
        "--lr",
        str(prof.lr),
        "--max-hours",
        str(args.max_hours),
    ]
    if prof.lora:
        argv.append("--lora")
    _say("$ " + " ".join(argv))
    rc = subprocess.call(argv)
    progress.record(paths, "train", "done" if rc == 0 else "failed", detail=f"exit {rc}", run=run)
    return rc


def cmd_all(args: argparse.Namespace) -> int:
    paths = _paths(args)
    steps = [
        ("setup", cmd_setup, {"profile": args.profile}),
        ("data", cmd_data, {"val_clips": args.val_clips, "test_clips": args.test_clips,
                            "seed": args.seed, "source": args.source, "force": args.force}),
        ("model", cmd_model, {}),
        ("baseline", cmd_baseline, {}),
        ("train", cmd_train, {"profile": args.profile, "steps": args.steps, "lr": args.lr,
                              "lora": args.lora, "max_hours": args.max_hours}),
        ("eval", cmd_eval, {"ckpt": args.ckpt, "name": None}),
        ("compare", cmd_compare, {"baseline": "baseline", "tuned": "tuned"}),
    ]
    for stage, fn, extra in steps:
        if args.resume and progress.status_of(paths, stage) == "done":
            _say(f"[skip] {stage} (done)")
            continue
        _say(f"\n=== {stage} ===")
        sub = argparse.Namespace(**{**vars(args), **extra})
        rc = fn(sub)
        if rc != 0:
            _say(f"\nstopped at '{stage}' (exit {rc}). Fix it, then: sheng-voiceclone all --resume")
            return rc
    _say("\nall stages done. Listen in " + str(paths.evals))
    return 0


def cmd_prompts(args: argparse.Namespace) -> int:
    for p in evals.eval_prompts(args.only):
        _say(f"{p['id']:<12} [{p['bucket']:<7}] {p['text']}")
    return 0


# ---------------------------------------------------------------------- parser
def _add_synth_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prompts", default=None, help="prompt ids or buckets: sheng,swahili,english,sheng_01")
    p.add_argument("--language", default="sw", help="Chatterbox language id (default sw)")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg-weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--force", action="store_true", help="re-synthesise even if WAVs exist")
    p.add_argument("--dry-run", action="store_true", help="plan only, no model, no GPU")


def _add_score_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--asr", action=argparse.BooleanOptionalAction, default=True,
                   help="score CER/WER with Whisper (default on)")
    p.add_argument("--sim", action=argparse.BooleanOptionalAction, default=True,
                   help="score speaker similarity with ECAPA (default on)")
    p.add_argument("--asr-model", default=evals.DEFAULT_ASR)
    p.add_argument("--spk-model", default=evals.DEFAULT_SPK)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sheng-voiceclone",
        description="Measure-first voice cloning for Sheng (Swahili + English) on Colab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Typical first run: setup -> data -> model -> baseline -> train -> eval -> compare",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="where am I, what is next")
    _common(s)
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("setup", help="environment + GPU profile")
    _common(s)
    s.add_argument("--profile", default="auto", choices=["auto", *cfg.PROFILES])
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("data", help="dataset -> 16 kHz WAVs, splits, reference clip")
    _common(s)
    s.add_argument("--source", default=None, help="use a local dir of wavs (+ .txt / transcripts.csv)")
    s.add_argument("--val-clips", type=int, default=4)
    s.add_argument("--test-clips", type=int, default=4)
    s.add_argument("--seed", type=int, default=1337)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("model", help="download the Chatterbox Multilingual snapshot")
    _common(s)
    s.set_defaults(fn=cmd_model)

    s = sub.add_parser("baseline", help="zero-shot clone + score (the number to beat)")
    _common(s)
    _add_synth_args(s)
    _add_score_args(s)
    s.set_defaults(fn=cmd_baseline)

    s = sub.add_parser("train", help="fine-tune T3 (delegates to finetune-chatterbox)")
    _common(s)
    s.add_argument("--profile", default="auto", choices=["auto", *cfg.PROFILES])
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--lr", type=float, default=None)
    s.add_argument("--lora", action="store_true", help="LoRA instead of full T3 fine-tune")
    s.add_argument("--max-hours", type=float, default=8.0)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("eval", help="score a fine-tuned checkpoint")
    _common(s)
    _add_synth_args(s)
    _add_score_args(s)
    s.add_argument("--ckpt", default=None, help="checkpoint file or dir (default: <run>/latest)")
    s.add_argument("--name", default=None, help="eval dir name (default: tuned_<step>)")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("score", help="re-score an existing eval dir")
    _common(s)
    _add_score_args(s)
    s.add_argument("--dir", required=True, help="directory name under evals/ (e.g. baseline)")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("compare", help="baseline vs tuned, with a verdict")
    _common(s)
    s.add_argument("--baseline", default="baseline")
    s.add_argument("--tuned", default="tuned")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("all", help="run every stage, resumable")
    _common(s)
    _add_synth_args(s)
    _add_score_args(s)
    s.add_argument("--profile", default="auto", choices=["auto", *cfg.PROFILES])
    s.add_argument("--source", default=None)
    s.add_argument("--val-clips", type=int, default=4)
    s.add_argument("--test-clips", type=int, default=4)
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--lr", type=float, default=None)
    s.add_argument("--lora", action="store_true")
    s.add_argument("--max-hours", type=float, default=8.0)
    s.add_argument("--ckpt", default=None)
    s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_all)

    s = sub.add_parser("prompts", help="print the Sheng eval prompt set")
    s.add_argument("--only", default=None)
    s.set_defaults(fn=cmd_prompts)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except CliError as exc:
        _say(f"ERROR: {exc}")
        return 1
    except SystemExit:
        raise
    except KeyboardInterrupt:
        _say("\ninterrupted - re-run with --resume")
        return 130
    except Exception as exc:  # one clean line beats a traceback on Colab
        _say(f"ERROR ({type(exc).__name__}): {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
