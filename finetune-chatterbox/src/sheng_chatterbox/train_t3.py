"""Fine-tune Chatterbox's T3 (text -> speech-token LM) on a JSONL manifest.

Only T3 is trained. ``ve`` (voice encoder), ``s3gen`` (token -> waveform vocoder)
and the S3 tokenizer are frozen, so the fine-tune can change *who* and *which
accent* the model speaks with, but cannot degrade the audio decoder.

Run by ``sheng-chatterbox train`` (which builds the argv), or by hand:

    python -m sheng_chatterbox.train_t3 \
        --train-manifest artifacts/manifests/train.jsonl \
        --pretrained-dir artifacts/pretrained/chatterbox \
        --save-dir artifacts/checkpoints/sheng_t3 \
        --steps 600

Checkpoint layout (one dir per save, plus ``latest``)::

    step_0000100/
      t3_cfg.safetensors      fp16 T3 state_dict - same keys as the official file
      training_state.json     step, losses, argv, timestamps
      run_config.json         resolved arguments (reproducibility)
    latest/                   copy of the newest step dir; what --resume loads
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

from .t3data import FeaturizeConfig, Featurizer, SpeechCollator, T3Dataset, endless

CKPT_NAME = "t3_cfg.safetensors"
STATE_NAME = "training_state.json"


# ----------------------------------------------------------------- arguments
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", default="")
    p.add_argument("--pretrained-dir", required=True, help="dir with ve/s3gen/tokenizer/conds + base T3")
    p.add_argument("--base-t3", default="t3_mtl23ls_v2.safetensors", help="T3 weights inside --pretrained-dir")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--cache-dir", default="")
    p.add_argument("--log-dir", default="")
    p.add_argument("--run", default="sheng_t3")

    p.add_argument("--model-variant", choices=["mtl", "english"], default="mtl")
    p.add_argument("--language-id", default="sw", help="text tokenizer language (mtl variant)")
    p.add_argument("--steps", type=int, default=600, help="optimizer steps (not micro-batches)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-text-len", type=int, default=512)
    p.add_argument("--max-speech-len", type=int, default=1024)
    p.add_argument("--cond-seconds", type=float, default=6.0)
    p.add_argument("--emotion-adv", type=float, default=0.5)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--valid-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--valid-batches", type=int, default=8)
    p.add_argument("--tensorboard", default="")
    p.add_argument("--resume", default="auto", help="auto | none | path to a step dir")
    p.add_argument("--dry-run", action="store_true", help="featurise + 1 forward/backward, then exit")
    p.add_argument("--device", default="cuda")
    return p


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------- model
def load_model(args: argparse.Namespace):
    """Load the base Chatterbox model and return it with T3 trainable."""
    import torch

    variant = args.model_variant
    if variant == "mtl":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as TTS

        model = TTS.from_local(args.pretrained_dir, args.device, t3_model=args.base_t3)
    else:
        from chatterbox.tts import ChatterboxTTS as TTS

        model = TTS.from_local(args.pretrained_dir, args.device)

    for module_name in ("ve", "s3gen"):
        module = getattr(model, module_name)
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    model.t3.train()
    for param in model.t3.parameters():
        param.requires_grad_(True)
    return model


def t3_device(model) -> "torch.device":
    return next(model.t3.parameters()).device


def amp_settings(dtype: str):
    """(autocast ctx factory, grad-scaler enabled)."""
    import torch

    if dtype == "fp32":
        return (lambda: torch.autocast("cuda", enabled=False)), False
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return (lambda: torch.autocast("cuda", dtype=torch_dtype)), dtype == "fp16"


def make_grad_scaler(enabled: bool):
    import torch

    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)  # type: ignore[attr-defined]


# ------------------------------------------------------------------ schedule
def lr_at(step: int, *, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine to 0 - a pure function of ``step``, so it resumes."""
    if warmup > 0 and step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / span))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------- checkpoints
def _to_fp16_state(state: dict[str, Any]) -> dict[str, Any]:
    import torch

    out = {}
    for key, value in state.items():
        if value.is_floating_point():
            out[key] = value.detach().to(torch.float16).cpu().contiguous()
        else:
            out[key] = value.detach().cpu().contiguous()
    return out


def save_checkpoint(
    model,
    args: argparse.Namespace,
    step: int,
    metrics: dict[str, Any],
    *,
    keep_last: int,
    counter: dict[str, int] | None = None,
) -> Path:
    """Write ``step_%07d/`` (+ refresh ``latest/``) and rotate old step dirs."""
    from safetensors.torch import save_file

    save_dir = Path(args.save_dir)
    step_dir = save_dir / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    save_file(_to_fp16_state(model.t3.state_dict()), str(step_dir / CKPT_NAME))

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    state = {
        "step": int(step),
        "created": now,
        "metrics": metrics,
        "argv": sys.argv[1:],
        "run": args.run,
        "model_variant": args.model_variant,
        "language_id": args.language_id,
        "steps_total": int(args.steps),
        "base_t3": args.base_t3,
        "pretrained_dir": str(args.pretrained_dir),
        "keep_last": int(keep_last),
    }
    (step_dir / STATE_NAME).write_text(json.dumps(state, indent=2), encoding="utf-8")
    (step_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    latest = save_dir / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for name in (CKPT_NAME, STATE_NAME):
        shutil.copy2(step_dir / name, latest / name)

    _rotate(save_dir, keep_last)
    count = counter or {}
    count["saves"] = count.get("saves", 0) + 1
    return step_dir


def _rotate(save_dir: Path, keep_last: int) -> None:
    steps = sorted(
        (p for p in save_dir.iterdir() if p.is_dir() and p.name.startswith("step_")),
        key=lambda p: p.name,
    )
    for old in steps[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(old, ignore_errors=True)


def resume_path(args: argparse.Namespace) -> Path | None:
    if args.resume == "none":
        return None
    if args.resume not in ("auto", ""):
        return Path(args.resume)
    latest = Path(args.save_dir) / "latest"
    return latest if (latest / CKPT_NAME).exists() else None


def load_into(model, path: Path) -> int:
    """Restore T3 weights (and the step counter) from a checkpoint dir."""
    from safetensors.torch import load_file

    state = load_file(str(Path(path) / CKPT_NAME))
    model.t3.load_state_dict(state, strict=True)
    try:
        meta = json.loads((Path(path) / STATE_NAME).read_text(encoding="utf-8"))
        return int(meta.get("step", 0))
    except (OSError, json.JSONDecodeError, ValueError):
        return 0


# -------------------------------------------------------------------- batches
def to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def batch_loss(model, batch: dict[str, Any]):
    """Forward T3 and return (total, text, speech)."""
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    cond = T3Cond(
        speaker_emb=batch["t3_cond_speaker_emb"],
        cond_prompt_speech_tokens=batch["t3_cond_prompt_speech_tokens"],
        cond_prompt_speech_emb=None,
        emotion_adv=batch["t3_cond_emotion_adv"],
    ).to(device=t3_device(model))

    loss_text, loss_speech = model.t3.loss(
        t3_cond=cond,
        text_tokens=batch["text_tokens"],
        text_token_lens=batch["text_token_lens"],
        speech_tokens=batch["speech_tokens"],
        speech_token_lens=batch["speech_token_lens"],
    )
    return loss_text + loss_speech, loss_text, loss_speech


# ------------------------------------------------------------------ data

def build_loaders(model, args: argparse.Namespace):
    import torch

    cfg = FeaturizeConfig(
        language_id=args.language_id,
        model_variant=args.model_variant,
        max_text_len=args.max_text_len,
        max_speech_len=args.max_speech_len,
        cond_seconds=args.cond_seconds,
        emotion_adv=args.emotion_adv,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    featurizer = Featurizer(model, cfg, device=args.device)
    collate = SpeechCollator(text_pad=0, speech_pad=0)

    train_ds = T3Dataset(Path(args.train_manifest), featurizer)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
        persistent_workers=bool(args.num_workers),
    )

    val_loader = None
    if args.val_manifest and Path(args.val_manifest).exists():
        val_ds = T3Dataset(Path(args.val_manifest), featurizer)
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
    return train_loader, val_loader, train_ds


def evaluate(model, val_loader, args: argparse.Namespace, max_batches: int) -> float | None:
    """Mean validation loss over at most ``max_batches`` batches (None = no val set)."""
    import torch

    if val_loader is None:
        return None
    model.t3.eval()
    autocast, _ = amp_settings(args.dtype)
    total, count = 0.0, 0
    with torch.no_grad(), autocast():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            if not batch:
                continue
            loss, _, _ = batch_loss(model, to_device(batch, t3_device(model)))
            total += float(loss.detach())
            count += 1
    model.t3.train()
    return total / count if count else None


# ------------------------------------------------------------------- loop

def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    start_time = time.time()

    log(f"[train] loading {args.model_variant} model from {args.pretrained_dir} ({args.base_t3})")
    model = load_model(args)
    device = t3_device(model)
    log(f"[train] device={device} dtype={args.dtype} steps={args.steps} "
        f"batch={args.batch_size}x{args.grad_accum}={args.batch_size * args.grad_accum}")

    train_loader, val_loader, train_ds = build_loaders(model, args)
    log(f"[train] {len(train_ds)} train clips, val={'yes' if val_loader else 'no'}")

    params = [p for p in model.t3.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    log(f"[train] trainable params: {n_params / 1e6:.1f}M (T3 only)")
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    step = 0
    ckpt = resume_path(args)
    if ckpt is not None:
        step = load_into(model, ckpt)
        log(f"[train] resumed from {ckpt} at step {step}")

    if args.dry_run:
        return _dry_run(model, train_loader, optimizer, args, step)

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(args.tensorboard)
            log(f"[train] tensorboard -> {args.tensorboard}")
        except Exception as exc:  # noqa: BLE001 - nice to have, never fatal
            log(f"[train] tensorboard disabled: {exc}")

    stop = {"flag": False, "reason": ""}

    def _request_stop(signum, _frame):
        stop["flag"] = True
        stop["reason"] = f"signal {signum}"
        log(f"[train] {stop['reason']} received - will checkpoint and exit after this step")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _request_stop)
        except (ValueError, OSError):
            pass  # not the main thread / unsupported platform

    autocast, use_scaler = amp_settings(args.dtype)
    scaler = make_grad_scaler(use_scaler)
    micro = 0
    empty_streak = 0
    running: dict[str, float] = {"loss": 0.0, "loss_text": 0.0, "loss_speech": 0.0}
    last_metrics: dict[str, Any] = {}
    best_val: float | None = None
    saves = {"saves": 0}
    steps_at_start = step

    optimizer.zero_grad(set_to_none=True)
    batches = endless(train_loader)

    for step in range(step, args.steps):
        lr = lr_at(step, base_lr=args.lr, warmup=args.warmup_steps, total=args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = next(batches)
        if not batch:
            empty_streak += 1
            if empty_streak >= 50:
                raise SystemExit("[train] 50 empty batches in a row - the manifest/WAVs are unusable")
            continue
        empty_streak = 0
        batch = to_device(batch, device)

        with autocast():
            loss, loss_text, loss_speech = batch_loss(model, batch)
            scaled = loss / args.grad_accum

        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        micro += 1
        running["loss"] += float(loss.detach())
        running["loss_text"] += float(loss_text.detach())
        running["loss_speech"] += float(loss_speech.detach())

        if micro % args.grad_accum != 0 and step != args.steps - 1:
            continue

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # accumulated over `micro` micro-batches since the previous update
        done = max(1, micro)
        mean_loss = running["loss"] / done
        mean_text = running["loss_text"] / done
        mean_speech = running["loss_speech"] / done
        last_metrics = {
            "step": step + 1,
            "loss": round(mean_loss, 4),
            "loss_text": round(mean_text, 4),
            "loss_speech": round(mean_speech, 4),
            "lr": lr,
        }
        running = {"loss": 0.0, "loss_text": 0.0, "loss_speech": 0.0}
        micro = 0

        if (step + 1) % args.log_interval == 0 or step == 0:
            elapsed = max(1e-6, time.time() - start_time)
            per_step = elapsed / max(1, step + 1 - steps_at_start)
            log(
                f"step={step + 1:07d} loss={mean_loss:.4f} loss_text={mean_text:.4f} "
                f"loss_speech={mean_speech:.4f} lr={lr:.8f} sec/step={per_step:.2f}"
            )
            if writer:
                writer.add_scalar("train/loss", mean_loss, step + 1)
                writer.add_scalar("train/loss_text", mean_text, step + 1)
                writer.add_scalar("train/loss_speech", mean_speech, step + 1)
                writer.add_scalar("train/lr", lr, step + 1)

        if args.valid_interval > 0 and (step + 1) % args.valid_interval == 0:
            val = evaluate(model, val_loader, args, args.valid_batches)
            if val is not None:
                best_val = val if best_val is None else min(best_val, val)
                log(f"val step={step + 1:07d} val_loss={val:.4f} best={best_val:.4f}")
                last_metrics["val_loss"] = round(val, 4)
                last_metrics["best_val"] = round(best_val, 4)
                if writer:
                    writer.add_scalar("val/loss", val, step + 1)

        if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
            path = save_checkpoint(model, args, step + 1, last_metrics, keep_last=args.keep_last, counter=saves)
            log(f"[train] saved {path}")

        if stop["flag"]:
            break

    final_step = step + 1
    if final_step > steps_at_start and (args.save_interval <= 0 or final_step % args.save_interval != 0):
        path = save_checkpoint(model, args, final_step, last_metrics, keep_last=args.keep_last, counter=saves)
        log(f"[train] saved final checkpoint {path}")

    summary = {
        "step": final_step,
        "steps_total": args.steps,
        "stopped": stop["reason"] or None,
        "last": last_metrics,
        "best_val": best_val,
        "saves": saves["saves"],
        "minutes": round((time.time() - start_time) / 60, 2),
        "failures": train_ds.failures[:10],
    }
    log(f"[train] finished {json.dumps({k: v for k, v in summary.items() if k != 'failures'})}")
    if writer:
        writer.close()
    return summary


def _dry_run(model, train_loader, optimizer, args: argparse.Namespace, step: int) -> dict[str, Any]:
    """One batch end to end: cheap smoke test before spending Colab minutes."""
    import torch

    batch = next(iter(train_loader))
    if not batch:
        raise SystemExit("[dry-run] no usable batch - check the manifest and the WAVs")
    batch = to_device(batch, t3_device(model))
    autocast, use_scaler = amp_settings(args.dtype)
    scaler = make_grad_scaler(use_scaler)
    with autocast():
        loss, loss_text, loss_speech = batch_loss(model, batch)
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    shapes = {k: tuple(v.shape) for k, v in batch.items()}
    log(f"[dry-run] batch shapes: {shapes}")
    log(
        f"[dry-run] loss={float(loss):.4f} text={float(loss_text):.4f} "
        f"speech={float(loss_speech):.4f} start_step={step}"
    )
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        log(f"[dry-run] peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return {"dry_run": True, "loss": round(float(loss), 4), "shapes": shapes}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not Path(args.pretrained_dir, args.base_t3).exists() and args.model_variant == "mtl":
        raise SystemExit(f"missing base T3 weights: {Path(args.pretrained_dir) / args.base_t3}")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
