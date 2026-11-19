"""Stage 5: single-speaker SFT of Qwen3-TTS-12Hz-0.6B-Base on Sheng.

This is ``finetuning/sft_12hz.py`` from QwenLM/Qwen3-TTS with the changes that
make it work, resumable and evaluable:

1. **the 0.6B fix** - text embeddings go through ``talker.text_projection``
   before being added to the codec embeddings (upstream crashes 2048 vs 1024,
   see ``embeds.py``);
2. **step-based checkpoints + resume** - upstream only writes at the end of an
   epoch and has no resume, which on a 12 h Colab VM means losing the run;
3. **a real LR schedule** - warmup + cosine instead of a constant 2e-5, which
   the issue tracker reports as "loss goes down, output is noise";
4. **validation loss** on a held-out split, written next to every checkpoint;
5. **no accelerate** dependency: plain torch, one GPU, gradient accumulation.

Everything argparse, so ``python -m sheng_qwentts.train --help`` works on Colab
without the CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch

from .batching import SeqIds, build_batch
from .embeds import assemble_inputs_embeds
from .model import load_tts, torch_dtype
from .paths import SPEAKER_FILE, STATE_FILE, WEIGHTS_FILE

SPEAKER_EMBEDDING_INDEX = 3000


# --------------------------------------------------------------------------- args
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Qwen3-TTS-12Hz-0.6B-Base (single speaker)")
    p.add_argument("--model-dir", required=True, help="local snapshot of the Base model")
    p.add_argument("--train-jsonl", required=True, help="JSONL with audio/text/ref_audio/audio_codes")
    p.add_argument("--val-jsonl", default="", help="optional held-out JSONL (same format)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run", default="sheng_06b")
    p.add_argument("--speaker-name", default="abel")
    p.add_argument("--steps", type=int, default=240, help="optimizer steps (not epochs)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--sub-talker-loss-weight", type=float, default=0.3)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=40)
    p.add_argument("--val-interval", type=int, default=40)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tensorboard", default="", help="dir for tensorboard scalars (empty = off)")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--max-hours", type=float, default=0.0, help="stop (and checkpoint) before the VM dies")
    p.add_argument("--smoke", action="store_true", help="2 steps, tiny batch, for a pipeline check")
    return p.parse_args(argv)


# ------------------------------------------------------------------------ helpers
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mel_from_wav(path: Path) -> torch.Tensor:
    """(1, T, 128) log-mel at 24 kHz - the speaker encoder / training input."""
    import numpy as np
    import soundfile as sf

    try:
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    except ImportError as exc:  # pragma: no cover - depends on the installed qwen-tts
        raise ImportError(
            "qwen-tts is not installed (or too old to expose mel_spectrogram): "
            "`pip install -U qwen-tts` (see `sheng-qwentts setup`)"
        ) from exc

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if int(sr) != 24000:
        raise ValueError(f"{path} is {sr} Hz; Qwen3-TTS mels are 24 kHz-only (run `sheng-qwentts data`)")
    with torch.inference_mode():
        mel = mel_spectrogram(
            torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
    return mel


def tokenize_text(processor, text: str) -> torch.Tensor:
    """Upstream's ``_build_assistant_text`` + ``[:, :-5]``."""
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    ids = processor(text=prompt, return_tensors="pt", padding=True)["input_ids"]
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids[:, :-5]


def build_items(rows: list[dict], processor, ref_mel: torch.Tensor) -> list[dict]:
    items = []
    for row in rows:
        codes = torch.tensor(row["audio_codes"], dtype=torch.long)
        items.append(
            {
                "text_ids": tokenize_text(processor, row["text"]),
                "audio_codes": codes,
                "ref_mel": ref_mel,
            }
        )
    return items


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    warm = max(1, args.warmup_steps)
    if step < warm:
        return args.lr * (step + 1) / warm
    total = max(warm + 1, args.steps)
    progress = min(1.0, (step - warm) / max(1, total - warm))
    floor = 0.1 * args.lr
    return floor + (args.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    steps = sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("step_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for old in steps[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)


def save_checkpoint(
    *,
    model,
    base_dir: Path,
    output_dir: Path,
    step: int,
    epoch: int,
    speaker_embedding: torch.Tensor,
    speaker_name: str,
    state_extra: dict,
) -> Path:
    """Write a self-contained HF model dir (config + weights + speech tokenizer)."""
    out = output_dir / f"step_{step:07d}"
    out.mkdir(parents=True, exist_ok=True)

    for src in base_dir.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(base_dir)
        if rel.as_posix() == WEIGHTS_FILE:
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    state_dict = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    for key in [k for k in state_dict if k.startswith("speaker_encoder") or k.startswith("speech_tokenizer")]:
        del state_dict[key]

    weight = state_dict["talker.model.codec_embedding.weight"]
    weight[SPEAKER_EMBEDDING_INDEX] = speaker_embedding[0].detach().to(weight.device).to(weight.dtype)

    from safetensors.torch import save_file

    save_file(state_dict, str(out / WEIGHTS_FILE))

    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["tts_model_type"] = "custom_voice"
    talker = cfg.setdefault("talker_config", {})
    talker["spk_id"] = {speaker_name: SPEAKER_EMBEDDING_INDEX}
    talker["spk_is_dialect"] = {speaker_name: False}
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    torch.save(speaker_embedding.detach().to("cpu"), out / SPEAKER_FILE)

    state = {"run": state_extra.get("run"), "step": step, "epoch": epoch, "speaker_name": speaker_name,
             "speaker_embedding_index": SPEAKER_EMBEDDING_INDEX, **state_extra}
    (out / STATE_FILE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "latest.txt").write_text(out.name + "\n", encoding="utf-8", newline="\n")
    return out


def speaker_encoder_from_base(base_dir: Path, model) -> bool:
    """Copy the pristine speaker-encoder weights into ``model`` (only those keys)."""
    from safetensors import safe_open

    path = base_dir / WEIGHTS_FILE
    if not path.is_file():
        return False
    prefix = "speaker_encoder."
    with safe_open(str(path), framework="pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    if not state:
        return False
    missing, unexpected = model.speaker_encoder.load_state_dict(state, strict=False)
    return not missing and not unexpected


def find_latest(output_dir: Path) -> Path | None:
    pointer = output_dir / "latest.txt"
    if pointer.is_file():
        target = output_dir / pointer.read_text(encoding="utf-8").strip()
        if (target / STATE_FILE).is_file() and (target / WEIGHTS_FILE).is_file():
            return target
    if not output_dir.is_dir():
        return None
    steps = sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("step_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for step in reversed(steps):
        if (step / STATE_FILE).is_file() and (step / WEIGHTS_FILE).is_file():
            return step
    return None


# --------------------------------------------------------------------- train step
def forward_loss(model, batch, speaker_embedding: torch.Tensor, sub_talker_weight: float):
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    embeds = assemble_inputs_embeds(
        talker=model.talker,
        input_text_ids=input_ids[:, :, 0],
        input_codec_ids=input_ids[:, :, 1],
        codec_ids=codec_ids,
        text_embedding_mask=batch["text_embedding_mask"],
        codec_embedding_mask=batch["codec_embedding_mask"],
        codec_mask=batch["codec_mask"],
        speaker_embedding=speaker_embedding,
    )
    outputs = model.talker(
        inputs_embeds=embeds[:, :-1, :],
        attention_mask=batch["attention_mask"][:, :-1],
        labels=batch["codec_0_labels"][:, 1:],
        output_hidden_states=True,
    )
    hidden = outputs.hidden_states[0][-1]
    talker_hidden = hidden[batch["codec_mask"][:, :-1]]
    talker_codecs = codec_ids[batch["codec_mask"]]
    _, sub_loss = model.talker.forward_sub_talker_finetune(talker_codecs, talker_hidden)
    return outputs.loss + sub_talker_weight * sub_loss, outputs.loss.detach(), sub_loss.detach()


@torch.no_grad()
def evaluate(model, items, ids, speaker_embedding, sub_talker_weight, batch_size) -> float:
    model.eval()
    total, n = 0.0, 0
    for start in range(0, len(items), batch_size):
        batch = build_batch(items[start : start + batch_size], ids)
        batch = {k: v.to(speaker_embedding.device) for k, v in batch.items()}
        loss, _, _ = forward_loss(model, batch, speaker_embedding, sub_talker_weight)
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(1, n)


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.smoke:
        args.steps, args.save_interval, args.val_interval = 2, 2, 1
        args.log_interval = 1
    if args.batch_size * args.grad_accum < 1:
        raise ValueError("batch_size * grad_accum must be >= 1")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    model_path = base_dir
    resume_from = find_latest(output_dir) if args.resume else None
    if resume_from is not None:
        state = json.loads((resume_from / STATE_FILE).read_text(encoding="utf-8"))
        start_step = int(state.get("step", 0))
        model_path = resume_from
        print(f"[train] resuming from {resume_from} at step {start_step}/{args.steps}", flush=True)

    tts = load_tts(model_path, dtype=args.dtype, attn=args.attn, device=args.device)
    model = tts.model
    model.train()

    rows = read_jsonl(Path(args.train_jsonl))
    if not rows:
        raise RuntimeError(f"{args.train_jsonl} is empty")
    if "audio_codes" not in rows[0]:
        raise RuntimeError("training JSONL has no audio_codes - run `sheng-qwentts codes`")
    ref_paths = {str(Path(r["ref_audio"]).resolve()) for r in rows}
    if len(ref_paths) != 1:
        raise RuntimeError(
            "every line must use the same ref_audio for single-speaker fine-tuning, got: "
            f"{sorted(ref_paths)[:3]}"
        )
    ref_mel = mel_from_wav(Path(next(iter(ref_paths))))
    items = build_items(rows, tts.processor, ref_mel)
    val_items = []
    if args.val_jsonl:
        val_rows = read_jsonl(Path(args.val_jsonl))
        if val_rows and "audio_codes" in val_rows[0]:
            val_items = build_items(val_rows, tts.processor, ref_mel)

    ids = SeqIds.from_config(model.config)

    device = torch.device(args.device)
    ref_mel = ref_mel.to(device).to(torch_dtype(args.dtype))
    speaker_file = Path(model_path) / SPEAKER_FILE
    if resume_from is not None and speaker_file.is_file():
        # exact vector the run started with - checkpoints do not carry the
        # speaker encoder, so recomputing from a checkpoint would give noise
        speaker_embedding = torch.load(speaker_file, map_location="cpu")
        print(f"[train] speaker embedding from {speaker_file.name}")
    else:
        if resume_from is not None and speaker_encoder_from_base(base_dir, model):
            print("[train] checkpoint has no speaker_embedding.pt - using the Base speaker encoder")
        with torch.no_grad():
            speaker_embedding = model.speaker_encoder(ref_mel)
    if speaker_embedding.dim() == 1:
        speaker_embedding = speaker_embedding.unsqueeze(0)
    speaker_embedding = speaker_embedding.detach().to(torch_dtype(args.dtype))

    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("talker."))
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    use_fp16 = args.dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    print(
        f"[train] {n_trainable / 1e6:.0f}M trainable params | {len(items)} train / {len(val_items)} val clips "
        f"| batch {args.batch_size} x accum {args.grad_accum} | steps {start_step}->{args.steps} | "
        f"{args.dtype} / {args.attn}",
        flush=True,
    )

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            Path(args.tensorboard).mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=args.tensorboard)
            print(f"[train] tensorboard -> {args.tensorboard}")
        except Exception as exc:  # tensorboard not installed
            print(f"[train] tensorboard disabled ({exc})")

    deadline = time.time() + args.max_hours * 3600 if args.max_hours > 0 else None
    rng = torch.Generator().manual_seed(args.seed + start_step)
    order: list[int] = []
    step = start_step
    epoch = 0
    rolling: list[float] = []
    started = time.time()

    while step < args.steps:
        if not order:
            order = torch.randperm(len(items), generator=rng).tolist()
            epoch += 1
        batch_idx, order = order[: args.batch_size], order[args.batch_size :]
        batch_items = [items[i] for i in batch_idx]
        # gradient accumulation: fill the rest of the effective batch
        for _ in range(args.grad_accum - 1):
            if not order:
                order = torch.randperm(len(items), generator=rng).tolist()
                epoch += 1
            batch_idx, order = order[: args.batch_size], order[args.batch_size :]
            batch_items += [items[i] for i in batch_idx]

        batch = build_batch(batch_items, ids)
        batch = {k: v.to(device) for k, v in batch.items()}

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)

        loss, lm_loss, sub_loss = forward_loss(model, batch, speaker_embedding, args.sub_talker_loss_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        rolling.append(float(loss.item()))
        rolling = rolling[-20:]
        if step % args.log_interval == 0 or step == args.steps:
            elapsed = time.time() - started
            print(
                f"[train] step {step}/{args.steps} loss {sum(rolling) / len(rolling):.4f} "
                f"(lm {float(lm_loss):.4f} sub {float(sub_loss):.4f}) "
                f"lr {lr_at(step, args):.2e} {elapsed / max(1, step - start_step):.1f}s/step",
                flush=True,
            )

        if writer is not None:
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("train/lm_loss", float(lm_loss), step)
            writer.add_scalar("train/sub_talker_loss", float(sub_loss), step)
            writer.add_scalar("train/lr", lr_at(step, args), step)

        if val_items and args.val_interval > 0 and step % args.val_interval == 0:
            val_loss = evaluate(model, val_items, ids, speaker_embedding,
                                args.sub_talker_loss_weight, args.batch_size)
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, step)
            print(f"[train] step {step} val loss {val_loss:.4f}", flush=True)

        if args.save_interval > 0 and step % args.save_interval == 0 and step < args.steps:
            out = save_checkpoint(
                model=model, base_dir=base_dir, output_dir=output_dir, step=step, epoch=epoch,
                speaker_embedding=speaker_embedding, speaker_name=args.speaker_name,
                state_extra={"train_jsonl": str(args.train_jsonl), "lr": args.lr,
                             "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                             "dtype": args.dtype, "samples_seen": step * args.batch_size * args.grad_accum},
            )
            prune_checkpoints(output_dir, args.keep_last)
            print(f"[train] saved {out}", flush=True)

        if deadline is not None and time.time() > deadline:
            print("[train] --max-hours reached, checkpointing and stopping", flush=True)
            break

    final = save_checkpoint(
        model=model, base_dir=base_dir, output_dir=output_dir, step=step, epoch=epoch,
        speaker_embedding=speaker_embedding, speaker_name=args.speaker_name,
        state_extra={"train_jsonl": str(args.train_jsonl), "lr": args.lr,
                     "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                     "dtype": args.dtype, "samples_seen": step * args.batch_size * args.grad_accum,
                     "final": True},
    )
    if writer is not None:
        writer.close()
    print(f"[train] done at step {step} -> {final}", flush=True)
    return {"steps": step, "checkpoint": str(final), "seconds": round(time.time() - started, 1)}


if __name__ == "__main__":
    main()
