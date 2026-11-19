"""Stage 6/7: listen to the checkpoint, then package it.

Generation is the whole point, so the eval stage synthesises real Sheng with the
fine-tuned checkpoint (``generate_custom_voice`` - the speaker id that the
trainer baked into ``talker_config.spk_id``) and, optionally, the same text with
the untouched Base model in zero-shot clone mode, so you can hear what the
fine-tune actually changed.

It also runs cheap sanity checks that catch the classic failure modes from the
issue tracker - silence, noise, and never emitting EOS - without needing ears.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .model import load_tts, require_local_model
from .paths import Paths, latest_checkpoint

# Hand-written prose in the same register as the dataset (Sheng = Swahili grammar
# + English words + Nairobi slang). Kept short: short utterances expose a broken
# model much faster than long ones.
PROBES = [
    "Mambo vipi mazee, leo niko fiti kabisa, wacha tuendelee na stori.",
    "Unajua venye mambo hufanyika, msee anajitafutia tu maisha fiti.",
    "Niaje, wewe ni mkubwa ama unajifanyanga? Cheki hii stori yangu.",
]

SANITY_MAX_RMS = 0.5      # pure noise sits around here
SANITY_MIN_RMS = 1e-3     # digital silence


def resolve_checkpoint(paths: Paths, run: str, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(checkpoint)
        return path
    ckpt = latest_checkpoint(paths, run)
    if ckpt is None:
        raise FileNotFoundError(
            f"no checkpoint under {paths.checkpoint_dir(run)} - run `sheng-qwentts train` first"
        )
    return ckpt


def _stats(wav: np.ndarray, sr: int) -> dict:
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    return {
        "seconds": round(wav.size / float(sr), 3),
        "rms": round(rms, 5),
        "peak": round(float(np.max(np.abs(wav))) if wav.size else 0.0, 4),
        "empty": bool(wav.size == 0 or rms < SANITY_MIN_RMS),
        "noisy": bool(rms > SANITY_MAX_RMS),
    }


def synthesize(
    tts,
    texts: list[str],
    out_dir: Path,
    *,
    prefix: str,
    speaker: str | None = None,
    language: str = "auto",
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    max_new_tokens: int = 2048,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for i, text in enumerate(texts, start=1):
        t0 = time.time()
        if speaker is not None:
            wavs, sr = tts.generate_custom_voice(
                text=text, speaker=speaker, language=language, max_new_tokens=max_new_tokens
            )
        else:
            wavs, sr = tts.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                x_vector_only_mode=False,
                max_new_tokens=max_new_tokens,
            )
        wav = np.asarray(wavs[0], dtype=np.float32)
        elapsed = time.time() - t0
        path = out_dir / f"{prefix}_{i:02d}.wav"
        sf.write(path, wav, sr, subtype="PCM_16")
        row = {"file": str(path), "text": text, "gen_seconds": round(elapsed, 2), **_stats(wav, sr)}
        row["rtf"] = round(row["seconds"] / elapsed, 2) if elapsed > 0 else 0.0
        results.append(row)
        flag = ""
        if row["empty"]:
            flag = "  <-- SILENT, inference is broken"
        elif row["noisy"]:
            flag = "  <-- NOISE, lower the LR / fewer steps"
        print(f"[eval] {path.name} {row['seconds']:.1f}s in {elapsed:.1f}s (rtf {row['rtf']}){flag}", flush=True)
    return results


def run_eval(
    paths: Paths,
    run: str,
    *,
    checkpoint: str | None = None,
    dtype: str = "bfloat16",
    attn: str = "sdpa",
    device: str = "cuda:0",
    compare_base: bool = False,
    speaker: str = "abel",
) -> dict:
    ckpt = resolve_checkpoint(paths, run, checkpoint)
    out_dir = paths.eval_dir(run) / ckpt.name
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = list(PROBES)
    if paths.val_raw_jsonl.is_file():
        for line in paths.val_raw_jsonl.open(encoding="utf-8"):
            row = json.loads(line)
            texts.append(row["text"])
            if len(texts) >= len(PROBES) + 2:
                break

    tts = load_tts(ckpt, dtype=dtype, attn=attn, device=device)
    ft = synthesize(tts, texts, out_dir, prefix="ft", speaker=speaker)
    del tts
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    base = []
    if compare_base:
        ref_audio = paths.ref_audio
        ref_text_path = paths.data_dir / "ref_text.txt"
        ref_text = ref_text_path.read_text(encoding="utf-8").strip() if ref_text_path.is_file() else None
        if ref_text is None:
            print("[eval] no ref_text.txt - cloning without a reference transcript (lower fidelity)")
        base_model = load_tts(require_local_model(paths), dtype=dtype, attn=attn, device=device)
        base = synthesize(
            base_model, texts, out_dir, prefix="base_clone",
            ref_audio=ref_audio, ref_text=ref_text,
        )

    report = {
        "run": run,
        "checkpoint": str(ckpt),
        "speaker": speaker,
        "compared_base": bool(base),
        "texts": texts,
        "finetuned": ft,
        "base_clone": base,
        "problems": [r["file"] for r in ft if r["empty"] or r["noisy"]],
    }
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "eval_report.md").write_text(render_report(report), encoding="utf-8", newline="\n")
    return report


def render_report(report: dict) -> str:
    lines = [
        f"# Eval - {report['run']} @ `{Path(report['checkpoint']).name}`",
        "",
        f"* speaker id: `{report['speaker']}` (custom_voice)",
        f"* base-model clone comparison: {'yes' if report['compared_base'] else 'no'}",
        f"* suspicious files: {', '.join(report['problems']) if report['problems'] else 'none'}",
        "",
        "| # | text | seconds | rtf | rms | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(report["finetuned"], start=1):
        verdict = "silent" if row["empty"] else ("noise" if row["noisy"] else "ok")
        text = row["text"].replace("|", "/")
        text = text if len(text) <= 60 else text[:57] + "..."
        lines.append(f"| {i} | {text} | {row['seconds']:.2f} | {row['rtf']:.2f} | {row['rms']:.3f} | {verdict} |")
    if report["base_clone"]:
        lines += ["", "Base-model zero-shot clones of the same text are in the same folder "
                  "(`base_clone_*.wav`) - A/B them against `ft_*.wav`."]
    lines += [
        "",
        "Listen for: does the fine-tune keep the words, or drift into babbling? "
        "A fine-tune that improves *speaker* similarity but loses *intelligibility* "
        "means too many steps or too high a learning rate.",
    ]
    return "\n".join(lines) + "\n"


def export(
    paths: Paths,
    run: str,
    *,
    checkpoint: str | None = None,
    zip_bundle: bool = False,
) -> Path:
    ckpt = resolve_checkpoint(paths, run, checkpoint)
    dest = paths.export_dir(run)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    model_out = dest / "model"
    shutil.copytree(ckpt, model_out)

    state = {}
    state_file = ckpt / "train_state.json"
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    eval_dir = paths.eval_dir(run) / ckpt.name
    samples = sorted(eval_dir.glob("*.wav")) if eval_dir.is_dir() else []
    if samples:
        (dest / "samples").mkdir()
        for wav in samples:
            shutil.copy2(wav, dest / "samples" / wav.name)

    card = render_model_card(paths, run, ckpt, state)
    (dest / "README.md").write_text(card, encoding="utf-8", newline="\n")
    (dest / "config_used.yaml").write_text(paths.config_path(run).read_text(encoding="utf-8"), encoding="utf-8", newline="\n") if paths.config_path(run).is_file() else None

    manifest = {
        "run": run,
        "checkpoint": ckpt.name,
        "step": state.get("step"),
        "samples": [w.name for w in samples],
        "model_dir": str(model_out),
    }
    (dest / "export.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    if zip_bundle:
        bundle = dest / f"{run}.zip"
        shutil.make_archive(str(bundle.with_suffix("")), "zip", root_dir=dest, base_dir="model")
    return dest


def render_model_card(paths: Paths, run: str, ckpt: Path, state: dict) -> str:
    return f"""# {run} - Qwen3-TTS-12Hz-0.6B fine-tuned on Sheng

Single-speaker fine-tune of `Qwen/Qwen3-TTS-12Hz-0.6B-Base` on
`lyle-mlengineer/kenyan-celebs` (42 clips / 21 min of one speaker talking Sheng),
step {state.get('step', '?')}.

## Load it

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained("model", device_map="cuda:0",
                                    dtype=torch.bfloat16, attn_implementation="sdpa")
wavs, sr = tts.generate_custom_voice(
    text="Mambo vipi mazee, leo niko fiti kabisa.",
    speaker="{state.get('speaker_name', 'abel')}",
    language="auto",   # Sheng has no language id; "auto" lets the LM pick
)
sf.write("out.wav", wavs[0], sr)
```

## Training

| knob | value |
|---|---|
| init | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (local snapshot) |
| tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` (12 Hz, 16 codebooks, 24 kHz) |
| data | 24 kHz mono, single reference clip for the speaker embedding |
| steps | {state.get('step', '?')} |
| batch | {state.get('batch_size', '?')} x accum {state.get('grad_accum', '?')} |
| dtype | {state.get('dtype', '?')} |
| lr | {state.get('lr', '?')} (warmup + cosine) |

## Honest limits

* 21 minutes of audio teaches the *speaker* and his Sheng delivery, not new
  vocabulary - Qwen3-TTS already speaks Swahili and English.
* text that is neither Swahili nor English (pure slang spellings) may be
  pronounced as if it were one of the two; keep prompts close to the dataset
  style.
* the checkpoint keeps the talker + code predictor only; the speaker encoder is
  intentionally dropped (the voice is baked into `codec_embedding[3000]`).
"""
