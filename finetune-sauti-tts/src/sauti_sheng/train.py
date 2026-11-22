"""Stage 4: fine-tune F5-TTS v1 Base on the Sheng clips.

This is a thin, explicit wrapper around upstream ``f5_tts.model.Trainer``. The
upstream ``finetune_cli.py`` cannot be used as-is because it resolves ``data/``
and ``ckpts/`` relative to the installed ``f5_tts`` package, and because it
copies whatever checkpoint you give it to ``pretrained_<name>.pt`` - which does
**not** stop F5 from resuming a training checkpoint (the branch is chosen by
keys; see ``model.py``).

Everything that matters is decided here and passed down explicitly:

* dataset = ``<artifacts>/data/f5/raw.arrow`` + ``duration.json`` (built by
  ``data.py`` from the 24 kHz WAVs);
* vocab = Sauti's own 2545-entry ``vocab.txt``;
* init = ``pretrained_sauti.pt`` (EMA only, no ``update`` key);
* batching = ``batch_size_type="sample"``, because every clip is 30 s and F5's
  frame sampler drops anything over its frame threshold;
* resume = whatever ``model_last.pt`` is already in the checkpoint dir.

Run ``python -m sauti_sheng.train --help`` for the knobs; the CLI in
``cli.py`` renders them from a GPU profile.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .compat import patch_torchaudio, seed_everything
from .config import DEFAULT_PROFILE, PROFILES, updates_for
from .paths import Paths, resolve_artifacts, resolve_scratch
from .progress import Progress


class TrainingTimeLimit(Exception):
    """Raised at a checkpoint boundary once ``--max-hours`` is used up."""


# --------------------------------------------------------------------- helpers
def link_or_copy(src: Path, dst: Path) -> str:
    """Hard-link/symlink when the filesystem allows it, copy otherwise.

    Drive's FUSE mount does not support symlinks, and a 1.35 GB copy per run is
    wasteful, so try cheap first and always end with a real file.
    """
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        pass
    shutil.copy2(src, dst)
    return "copy"


def load_f5_dataset(dataset_dir: Path, limit: int | None = None):
    """``raw.arrow`` + ``duration.json`` -> F5 ``CustomDataset``."""
    from datasets import Dataset as HFDataset
    from f5_tts.model.dataset import CustomDataset

    from .model import MEL_KWARGS

    raw = dataset_dir / "raw.arrow"
    duration_json = dataset_dir / "duration.json"
    for path in (raw, duration_json):
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing - run `sauti-sheng data` first")

    data = HFDataset.from_file(str(raw))
    durations = json.loads(duration_json.read_text(encoding="utf-8"))["duration"]
    if len(durations) != len(data):
        raise RuntimeError(f"{raw} has {len(data)} rows but {duration_json} has {len(durations)}")

    if limit is not None and limit < len(data):
        data = data.select(range(limit))
        durations = durations[:limit]

    return CustomDataset(data, durations=durations, **MEL_KWARGS)


def install_deadline(trainer, hours: float) -> None:
    """Stop cleanly at the first checkpoint boundary after ``hours``.

    F5's loop has no time budget. Hooking ``save_checkpoint`` keeps the stop on
    an update boundary (so the optimizer/scheduler state is consistent) instead
    of killing the process mid-step.
    """
    if hours <= 0:
        return
    deadline = time.monotonic() + hours * 3600.0
    original = trainer.save_checkpoint

    def save_checkpoint(update, last: bool = False):
        original(update, last=last)
        if not last and time.monotonic() >= deadline:
            original(update, last=True)
            raise TrainingTimeLimit(f"max-hours reached at update {update}")

    trainer.save_checkpoint = save_checkpoint


# ---------------------------------------------------------------------- parser
# Defaults mirror the default runtime profile (Colab T4) so that running this
# module directly is safe on a 16 GB card.
_DEFAULT = PROFILES[DEFAULT_PROFILE]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sauti_sheng.train", description=__doc__)
    p.add_argument("--artifacts", default=None)
    p.add_argument("--scratch", default=None)
    p.add_argument("--run", default="sheng_abel")
    p.add_argument("--dataset-dir", default=None, help="default: <artifacts>/data/f5")
    p.add_argument("--vocab", default=None, help="default: <artifacts>/data/f5/vocab.txt")
    p.add_argument("--checkpoint-dir", default=None, help="default: <artifacts>/checkpoints/<run>")
    p.add_argument("--pretrain", default=None, help="default: <artifacts>/pretrained/sauti-tts/sauti_ema.pt")

    p.add_argument("--epochs", type=int, default=_DEFAULT.epochs)
    p.add_argument("--batch-size", type=int, default=_DEFAULT.batch_size, help="samples per forward pass")
    p.add_argument("--grad-accum", type=int, default=_DEFAULT.grad_accum)
    p.add_argument("--lr", type=float, default=_DEFAULT.learning_rate)
    p.add_argument("--warmup", type=int, default=_DEFAULT.warmup_updates, help="warmup optimizer updates")
    p.add_argument("--save-per-updates", type=int, default=_DEFAULT.save_per_updates)
    p.add_argument("--last-per-updates", type=int, default=_DEFAULT.last_per_updates)
    p.add_argument("--keep-last", type=int, default=_DEFAULT.keep_last)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--mixed-precision", default=_DEFAULT.mixed_precision, choices=["bf16", "fp16", "no"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--limit", type=int, default=None, help="only the first N clips (debugging)")
    p.add_argument("--max-hours", type=float, default=0.0, help="stop at the next checkpoint after N hours")
    p.add_argument("--fresh", action="store_true", help="delete model_*.pt and start from the pretrained init")
    p.add_argument("--smoke", action="store_true", help="1 epoch, 4 clips, 1 sample per step: wiring only")
    p.add_argument("--log-samples", action="store_true", help="decode a sample every save (needs GPU time)")
    return p


def apply_smoke(args) -> None:
    if not args.smoke:
        return
    args.epochs = 1
    args.limit = args.limit or 4
    args.batch_size = 1
    args.grad_accum = 1
    args.warmup = 1
    args.save_per_updates = 1
    args.last_per_updates = 1
    args.keep_last = 1
    args.max_hours = 0.0
    print("[smoke] 1 epoch x 4 clips x 1 sample/step")


def clear_checkpoints(checkpoint_dir: Path) -> list[str]:
    removed = []
    for path in sorted(checkpoint_dir.glob("model_*.pt")):
        if path.name.startswith("pretrained_"):
            continue
        path.unlink()
        removed.append(path.name)
    return removed


# ------------------------------------------------------------------------ main
def run(args) -> int:
    paths = Paths(resolve_artifacts(args.artifacts), scratch=resolve_scratch(args.scratch)).ensure()
    apply_smoke(args)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else paths.f5_dataset_dir
    vocab = Path(args.vocab) if args.vocab else paths.f5_vocab
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else paths.checkpoint_dir(args.run)
    pretrain = Path(args.pretrain) if args.pretrain else paths.sauti_ema_ckpt
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not pretrain.is_file():
        raise SystemExit(f"missing pretrained checkpoint {pretrain} - run `sauti-sheng model` first")
    if not vocab.is_file():
        raise SystemExit(f"missing vocab {vocab} - run `sauti-sheng data` first")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: F5 fine-tuning is GPU-only (Colab T4/L4/A100)")
    device_name = torch.cuda.get_device_name(0)

    patch_torchaudio()
    seed_everything(args.seed)
    # TF32 is Ampere+; a no-op on the T4 but free speed on L4/A100.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        print(
            "[train] this GPU has no bf16 support (T4/Turing); the t4 default uses fp16 - "
            "add --mixed-precision fp16 to avoid an accelerate failure"
        )

    if args.fresh:
        removed = clear_checkpoints(checkpoint_dir)
        if removed:
            print(f"[train] --fresh removed {len(removed)} checkpoint(s)")

    how = link_or_copy(pretrain, paths.pretrained_pointer(args.run))
    print(f"[train] init: {pretrain} -> {paths.pretrained_pointer(args.run)} ({how})")

    dataset = load_f5_dataset(dataset_dir, limit=args.limit)
    n = len(dataset)
    per_epoch = updates_for(n, args.batch_size, args.grad_accum)
    total = per_epoch * args.epochs
    print(
        f"[train] {n} clips | batch {args.batch_size} x accum {args.grad_accum} "
        f"= {args.batch_size * args.grad_accum} samples/update | "
        f"{per_epoch} updates/epoch x {args.epochs} = {total} updates"
    )
    print(f"[train] gpu {device_name} | {args.mixed_precision} | lr {args.lr} | ckpt {checkpoint_dir}")

    from f5_tts.model import Trainer

    from .model import build_cfm
    from .text import f5_text_identity

    changed = sum(1 for i in range(n) if f5_text_identity(dataset.data[i]["text"]) is False)
    if changed:
        print(
            f"[train] WARNING: {changed}/{n} transcripts are altered by F5's "
            "convert_char_to_pinyin but are stored verbatim; training and inference "
            "would tokenise them differently"
        )

    model, vocab_size = build_cfm(vocab)
    print(f"[train] vocab {vocab_size}")

    accelerate_kwargs = {} if args.mixed_precision == "no" else {"mixed_precision": args.mixed_precision}
    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_warmup_updates=args.warmup,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last,
        checkpoint_path=str(checkpoint_dir),
        batch_size_per_gpu=args.batch_size,
        batch_size_type="sample",
        max_samples=args.batch_size,
        grad_accumulation_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        logger=None,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        accelerate_kwargs=accelerate_kwargs,
    )
    install_deadline(trainer, args.max_hours)

    progress = Progress(paths, args.run)
    progress.note(
        "train",
        {
            "clips": n,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "updates": total,
            "learning_rate": args.lr,
            "mixed_precision": args.mixed_precision,
            "gpu": device_name,
            "smoke": args.smoke,
        },
    )

    try:
        trainer.train(dataset, num_workers=max(1, args.num_workers), resumable_with_seed=args.seed)
    except TrainingTimeLimit as exc:
        print(f"[train] {exc}")
        return 0

    print(f"[train] done: {paths.latest_checkpoint(args.run)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
