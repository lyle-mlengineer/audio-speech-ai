"""Evaluate a fine-tuned T3 and package it for reuse.

Two jobs:

* ``eval``   - generate the same Sheng prompts from (a) the base model and (b)
  the checkpoint, so you can A/B them, and score duration/RTF/similarity.
* ``export`` - build a self-contained model directory (base files + fine-tuned
  ``t3_cfg.safetensors``) that ``ChatterboxMultilingualTTS.from_local`` loads
  directly, with a README, a load snippet and the eval samples.

Chatterbox checkpoints are *T3-only*: ``ve.pt`` (speaker encoder), ``s3gen.pt``
(vocoder) and the tokenizer stay exactly as shipped. So "exporting" is copying
those files next to the new T3 weights - no conversion step, no custom format.
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from .paths import (
    LANGUAGE_ID,
    MTL_T3_MODEL,
    PRETRAINED_FILES,
    Paths,
    checkpoint_step,
    latest_checkpoint,
)

CKPT_NAME = "t3_cfg.safetensors"

# Sheng: Swahili matrix language, English embedded. Keep prompts short - the
# model degrades on long texts, and long clips are what overfitting looks like.
EVAL_PROMPTS: dict[str, str] = {
    "sheng_casual": "Sasa wewe, unajua vile mambo iko. Let me tell you, hii ni serious.",
    "sheng_swahili": "Nilikuwa naenda kwa duka jana, nikapata rafiki yangu wa zamani.",
    "sheng_english": "Honestly speaking, hiyo meeting ilikuwa too long, but the budget is okay.",
    "unseen_vocab": "Wakati wa kubadilisha passport, lazima ulete documents zote mpya.",
    "long_form": (
        "Habari yako? Leo nataka kukuambia story ya kazi. I was working at the shop, "
        "na customer mmoja alikuja na shida. Tulikaa chini tukazungumza, and finally "
        "tukapata solution. That is why I love this job."
    ),
}


# ------------------------------------------------------------------- helpers
def resolve_step(paths: Paths, run: str, step: str | None) -> Path | None:
    """``latest`` (default), ``best`` (= latest), or a step number."""
    if step in (None, "", "latest", "best"):
        return latest_checkpoint(paths, run)
    ckpt = paths.step_dir(run, int(step))
    return ckpt if (ckpt / CKPT_NAME).exists() else None


def require_mtl() -> None:
    try:
        import chatterbox.mtl_tts  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "chatterbox is not importable - run `sheng-chatterbox env` first "
            f"(and restart the runtime if it just installed torch). Original error: {exc}"
        ) from exc


def load_finetuned(paths: Paths, ckpt: Path, device: str = "cuda"):
    """Load ``ve``/``s3gen``/tokenizer from the base dir + T3 from the checkpoint."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model = ChatterboxMultilingualTTS.from_local(
        str(paths.pretrained_dir), device, t3_model=MTL_T3_MODEL
    )
    from safetensors.torch import load_file

    state = load_file(str(Path(ckpt) / CKPT_NAME))
    model.t3.load_state_dict(state, strict=True)
    model.t3.to(device).eval()
    return model


def load_base(paths: Paths, device: str = "cuda"):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS.from_local(str(paths.pretrained_dir), device, t3_model=MTL_T3_MODEL)


def reference_clip(paths: Paths) -> Path:
    """Held-out clip if there is one, else the first training clip."""
    from . import manifest as M

    for manifest in (paths.val_manifest, paths.train_manifest):
        if not manifest.exists():
            continue
        rows = M.read_manifest(manifest)
        if rows:
            audio = Path(rows[0]["audio"])
            return audio if audio.is_absolute() else manifest.parent / audio
    raise FileNotFoundError("no manifest with audio found - run `sheng-chatterbox data`")


def _generate(model, text: str, ref_wav: Path, language_id: str, out_path: Path, **kw) -> dict[str, Any]:
    import soundfile as sf
    import torch

    model.prepare_conditionals(str(ref_wav), exaggeration=kw.pop("exaggeration", 0.5))
    started = time.time()
    with torch.inference_mode():
        wav = model.generate(text, language_id=language_id, **kw)
    elapsed = time.time() - started
    arr = wav.squeeze(0).detach().cpu().numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), arr, model.sr)
    audio_seconds = len(arr) / float(model.sr)
    return {
        "file": str(out_path),
        "audio_seconds": round(audio_seconds, 2),
        "wall_seconds": round(elapsed, 2),
        "rtf": round(elapsed / audio_seconds, 3) if audio_seconds else None,
        "ref": str(ref_wav),
    }


# ---------------------------------------------------------------------- eval
def evaluate(
    paths: Paths,
    run: str,
    *,
    step: str | None = None,
    ab: bool = True,
    language_id: str = LANGUAGE_ID,
    prompts: dict[str, str] | None = None,
    log=print,
) -> dict[str, Any]:
    require_mtl()
    ckpt = resolve_step(paths, run, step)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {paths.checkpoint_dir(run)} - train first")

    ref = reference_clip(paths)
    prompts = prompts or EVAL_PROMPTS
    out_dir = paths.eval_dir(run) / (ckpt.name if ckpt.name != "latest" else "latest")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[eval] checkpoint {ckpt} -> {out_dir}")
    log(f"[eval] reference clip {ref}")

    variants: list[tuple[str, Any]] = []
    if ab:
        log("[eval] loading base model for A/B")
        variants.append(("base", load_base(paths)))
    log("[eval] loading fine-tuned model")
    variants.append((ckpt.name, load_finetuned(paths, ckpt)))

    results: dict[str, Any] = {
        "run": run,
        "checkpoint": str(ckpt),
        "step": checkpoint_step(ckpt),
        "language_id": language_id,
        "reference": str(ref),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompts": {},
        "variants": [name for name, _ in variants],
    }

    for name, model in variants:
        results["prompts"].setdefault(name, {})
        for key, text in prompts.items():
            wav_path = out_dir / f"{name}__{key}.wav"
            log(f"[eval] {name}: {key} ({len(text)} chars)")
            try:
                info = _generate(model, text, ref, language_id, wav_path)
            except Exception as exc:  # noqa: BLE001 - one prompt must not kill the eval
                log(f"[eval] {name}/{key} failed: {exc}")
                info = {"error": str(exc)}
            info["text"] = text
            results["prompts"][name][key] = info
        del model

    (out_dir / "eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "eval.md").write_text(_render_eval_md(results), encoding="utf-8")
    log(f"[eval] wrote {out_dir / 'eval.md'} - listen to the wavs side by side")
    return results


def _render_eval_md(results: dict[str, Any]) -> str:
    lines = [
        f"# Eval - `{results['run']}` step {results.get('step')}",
        "",
        f"- checkpoint: `{results['checkpoint']}`",
        f"- reference clip: `{results['reference']}`",
        f"- language id: `{results['language_id']}`",
        f"- created: {results['created']}",
        "",
        "| prompt | variant | audio s | RTF | file |",
        "|---|---|---|---|---|",
    ]
    for variant, per_prompt in results["prompts"].items():
        for key, info in per_prompt.items():
            if "error" in info:
                lines.append(f"| {key} | {variant} | - | - | ERROR: {info['error']} |")
            else:
                lines.append(
                    f"| {key} | {variant} | {info.get('audio_seconds')} | {info.get('rtf')} | "
                    f"`{Path(info['file']).name}` |"
                )
    lines += [
        "",
        "**How to judge:** listen twice. First with the transcript visible, then "
        "without - the fine-tune is good when *new* Sheng text is intelligible and the "
        "voice sounds like abel. If it only sounds right on training sentences, it is "
        "overfitting: take an earlier checkpoint.",
    ]
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- export
def export(
    paths: Paths,
    run: str,
    *,
    step: str | None = None,
    with_samples: bool = True,
    zip_it: bool = True,
    log=print,
) -> Path:
    """Self-contained model dir (base files + fine-tuned t3_cfg.safetensors)."""
    ckpt = resolve_step(paths, run, step)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {paths.checkpoint_dir(run)} - train first")

    target = paths.export_dir(run) / (ckpt.name if ckpt.name != "latest" else "latest")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    for name in PRETRAINED_FILES:
        if name == MTL_T3_MODEL:
            continue  # replaced by the fine-tuned weights
        src = paths.pretrained_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing base file {src} - run `sheng-chatterbox model`")
        shutil.copy2(src, target / name)

    shutil.copy2(ckpt / CKPT_NAME, target / CKPT_NAME)
    step_num = checkpoint_step(ckpt)
    if step_num is None:
        state = ckpt / "training_state.json"
        if state.exists():
            try:
                step_num = json.loads(state.read_text(encoding="utf-8")).get("step")
            except json.JSONDecodeError:
                step_num = None

    (target / "README.md").write_text(_export_readme(run, step_num), encoding="utf-8")
    (target / "load_chatterbox_sheng.py").write_text(_export_snippet(), encoding="utf-8")

    if with_samples:
        eval_root = paths.eval_dir(run)
        candidates = sorted(eval_root.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            dst = target / "samples"
            dst.mkdir(exist_ok=True)
            for wav in candidates[0].glob("*.wav"):
                shutil.copy2(wav, dst / wav.name)
            log(f"[export] samples from {candidates[0].name}")

    meta = {
        "run": run,
        "step": step_num,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": "ResembleAI/chatterbox",
        "t3_file": CKPT_NAME,
        "language_ids": ["sw", "en"],
        "load": "ChatterboxMultilingualTTS.from_local(dir, device, t3_model='t3_cfg.safetensors')",
    }
    (target / "export.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"[export] wrote {target}")

    if zip_it:
        archive = shutil.make_archive(str(target), "zip", root_dir=str(target))
        log(f"[export] zipped -> {archive}")
    return target


def _export_readme(run: str, step: int | None) -> str:
    return f"""# Chatterbox x Sheng - T3 fine-tune (`{run}`, step {step})

Fine-tuned `T3` of `ResembleAI/chatterbox` (multilingual), trained on ~21 min of
Sheng (Swahili + English code-switching) speech from the `lyle-mlengineer/kenyan-celebs`
dataset. Only T3 was trained; the voice encoder, the S3 tokenizer and the s3gen
vocoder are the originals, so audio quality is unchanged.

## Load

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_local(
    "path/to/this/dir", device="cuda", t3_model="t3_cfg.safetensors"
)
model.prepare_conditionals("some_sheng_clip.wav")   # ~6 s of the target voice
wav = model.generate("Sasa wewe, unajua vile mambo iko.", language_id="sw")
```

## Files

| file | what |
|---|---|
| `t3_cfg.safetensors` | the fine-tuned T3 (fp16) - the only file that is new |
| `ve.pt` | voice encoder (speaker embedding) |
| `s3gen.pt` | s3gen vocoder (speech tokens -> 24 kHz audio) |
| `grapheme_mtl_merged_expanded_v1.json` | multilingual text tokenizer |
| `conds.pt` | built-in voice conditioning (optional) |
| `samples/` | A/B renders from `sheng-chatterbox eval --ab` |

The base model's `t3_mtl23ls_v2.safetensors` is deliberately **not** included:
`from_local(..., t3_model="t3_cfg.safetensors")` loads only the file you name.

## Notes

- Speak `sw` for Swahili/Sheng text; `en` still works but is not what was tuned.
- Use the same `exaggeration=0.5` used in training, or the voice shifts.
- fp16 weights: load on GPU. On CPU, cast with `.float()`.
"""


def _export_snippet() -> str:
    return '''"""Minimal load + generate for the exported Sheng checkpoint."""
import sys
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def main(out_dir: str, text: str, prompt_wav: str, out_wav: str = "out.wav") -> None:
    model = ChatterboxMultilingualTTS.from_local(out_dir, "cuda", t3_model="t3_cfg.safetensors")
    model.prepare_conditionals(prompt_wav, exaggeration=0.5)
    wav = model.generate(text, language_id="sw", exaggeration=0.5, cfg_weight=0.5, temperature=0.8)
    import torchaudio
    torchaudio.save(out_wav, wav.cpu(), model.sr)
    print(f"wrote {out_wav}")

if __name__ == "__main__":
    main(*sys.argv[1:])
'''


def zip_dir(path: Path) -> Path:
    archive = shutil.make_archive(str(path), "zip", root_dir=str(path))
    return Path(archive)


def unzip(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    return dest
