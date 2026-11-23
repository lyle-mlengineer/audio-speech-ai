"""Stage 6: does the fine-tune actually speak Sheng, or just make noise?

Three hand-written probes (Swahili + English the way the dataset talks) and two
held-out dataset transcripts are synthesised with the fine-tuned checkpoint.
With ``--compare-base`` the untouched Sauti release is run on the same prompts,
so you hear *what changed* rather than only what the new voice sounds like.

The checks are crude on purpose - they catch the failure modes that actually
happen after a short fine-tune (silence, runaway length, clipping/noise) before
you waste time listening to every file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .infer import NFE_STEP, load_reference, synthesise, write_wav
from .model import load_for_inference
from .paths import Paths
from .text import SHENG_PROBES, truncate_for_eval

FRAME_SECONDS = 0.02
SILENCE_RMS = 1e-4


# -------------------------------------------------------------------- metrics
def metrics(wave: np.ndarray, sample_rate: int = 24000) -> dict:
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if wave.size == 0:
        return {"seconds": 0.0, "rms": 0.0, "peak": 0.0, "silent_fraction": 1.0, "nan": False}
    frame = max(1, int(sample_rate * FRAME_SECONDS))
    usable = (wave.size // frame) * frame
    frames = wave[:usable].reshape(-1, frame) if usable else wave.reshape(1, -1)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return {
        "seconds": float(wave.size / sample_rate),
        "rms": float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))),
        "peak": float(np.max(np.abs(wave))),
        "silent_fraction": float(np.mean(frame_rms < SILENCE_RMS)),
        "nan": bool(np.isnan(wave).any() or np.isinf(wave).any()),
    }


def verdict(m: dict, max_seconds: float = 30.0) -> tuple[bool, str]:
    if m["nan"]:
        return False, "NaN/Inf in output - inference is broken"
    if m["seconds"] < 0.8:
        return False, "too short (< 0.8 s) - EOS fired immediately"
    if m["rms"] < 1e-3 or m["silent_fraction"] > 0.95:
        return False, "silent - check the checkpoint/vocoder"
    if m["seconds"] > max_seconds * 1.5:
        return False, "runaway length - EOS forgotten (LR too high / too many steps)"
    if m["peak"] >= 0.999 and m["rms"] > 0.4:
        return False, "clipping / noise - likely LR too high"
    return True, "ok"


# -------------------------------------------------------------------- prompts
def _held_out_prompts(paths: Paths, limit: int = 2) -> list[dict]:
    if not paths.val_csv.is_file():
        return []
    rows = []
    with open(paths.val_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="|"):
            text = truncate_for_eval(row.get("text", ""))
            if text:
                rows.append({"id": f"heldout_{len(rows) + 1:02d}", "text": text, "source": "val"})
            if len(rows) >= limit:
                break
    return rows


def prompts(paths: Paths) -> list[dict]:
    items = [
        {"id": f"ft_{i + 1:02d}", "text": text, "source": "probe"}
        for i, text in enumerate(SHENG_PROBES)
    ]
    return items + _held_out_prompts(paths)


# ----------------------------------------------------------------------- run
def _generate_set(
    paths: Paths,
    checkpoint: Path,
    use_ema: bool,
    items: list[dict],
    out_dir: Path,
    *,
    prefix: str,
    device: str,
    nfe_step: int,
    max_seconds: float,
) -> list[dict]:
    ref_audio, ref_rate, ref_text = load_reference(paths)
    model, _ = load_for_inference(paths, checkpoint, use_ema=use_ema, device=device)

    from .infer import load_vocoder

    vocoder = load_vocoder(device)
    results = []
    for item in items:
        wave = synthesise(
            model,
            vocoder,
            ref_audio,
            ref_text,
            item["text"],
            ref_sample_rate=ref_rate,
            nfe_step=nfe_step,
            device=device,
        )
        path = write_wav(out_dir / f"{prefix}_{item['id']}.wav", wave)
        m = metrics(wave)
        ok, reason = verdict(m, max_seconds=max_seconds)
        results.append(
            {
                "id": item["id"],
                "source": item["source"],
                "text": item["text"],
                "file": str(path),
                **m,
                "ok": ok,
                "verdict": reason,
            }
        )
        print(f"[eval] {path.name}: {m['seconds']:.1f}s rms={m['rms']:.3f} -> {reason}")

    del model, vocoder
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return results


def _write_report(dest: Path, checkpoint: Path, use_ema: bool, vocab_size: int | None,
                  finetuned: list[dict], base: list[dict]) -> None:
    lines = [
        "# Eval report",
        "",
        f"- checkpoint: `{checkpoint}` (use_ema={use_ema})",
        f"- vocab: {vocab_size} entries",
        "",
        "## Fine-tuned",
        "",
        "| file | source | seconds | rms | peak | silent | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in finetuned:
        lines.append(
            f"| `{Path(row['file']).name}` | {row['source']} | {row['seconds']:.2f} | "
            f"{row['rms']:.3f} | {row['peak']:.3f} | {row['silent_fraction'] * 100:.0f}% | "
            f"{'ok' if row['ok'] else '**' + row['verdict'] + '**'} |"
        )
    if base:
        lines += [
            "",
            "## Untouched Sauti (same prompts, same reference)",
            "",
            "| file | source | seconds | rms | peak | silent | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in base:
            lines.append(
                f"| `{Path(row['file']).name}` | {row['source']} | {row['seconds']:.2f} | "
                f"{row['rms']:.3f} | {row['peak']:.3f} | {row['silent_fraction'] * 100:.0f}% | "
                f"{'ok' if row['ok'] else '**' + row['verdict'] + '**'} |"
            )
        lines += [
            "",
            (
                "Listen to the `base_*` and `ft_*` pairs. A good fine-tune keeps the"
                "timbre and improves the Sheng delivery; a bad one is quieter, noisier,"
                "or forgets to stop."
            ),
        ]
    lines += [
        "",
        "## Prompts",
        "",
    ]
    for row in finetuned:
        lines.append(f"- `{row['id']}` ({row['source']}): {row['text']}")
    lines.append("")
    dest.write_text("\n".join(lines), encoding="utf-8")


def run_eval(
    paths: Paths,
    run: str,
    *,
    checkpoint: Path | str | None = None,
    use_ema: bool = False,
    compare_base: bool = True,
    device: str = "cuda",
    nfe_step: int = NFE_STEP,
    max_seconds: float = 30.0,
) -> dict:
    checkpoint = Path(checkpoint or paths.latest_checkpoint(run))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} missing - train first, or pass --checkpoint")

    items = prompts(paths)
    if not items:
        raise RuntimeError("no prompts to evaluate")

    dest = paths.eval_dir(run) / checkpoint.stem
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "prompts.json").write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")

    finetuned = _generate_set(
        paths, checkpoint, use_ema, items, dest,
        prefix="ft", device=device, nfe_step=nfe_step, max_seconds=max_seconds,
    )
    base: list[dict] = []
    if compare_base:
        if paths.sauti_ema_ckpt.is_file():
            base = _generate_set(
                paths, paths.sauti_ema_ckpt, True, items, dest,
                prefix="base", device=device, nfe_step=nfe_step, max_seconds=max_seconds,
            )
        else:
            print("[eval] --compare-base requested but the Sauti checkpoint is missing; skipping")

    vocab_size = None
    try:
        from .text import read_vocab

        vocab_size = len(read_vocab(paths.sauti_vocab))
    except Exception:
        pass
    _write_report(dest, checkpoint, use_ema, vocab_size, finetuned, base)

    problems = [row["file"] for row in finetuned + base if not row["ok"]]
    return {
        "dir": str(dest),
        "report": str(dest / "eval_report.md"),
        "finetuned": finetuned,
        "base": base,
        "problems": problems,
    }
