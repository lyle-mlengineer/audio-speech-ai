"""Evaluation + export for the Sheng LoRA.

Eval: synthesize a fixed set of Sheng / code-switched prompts with one or more
checkpoints, optionally A/B against LoRA-disabled base output, and write wavs +
``report.md`` under ``artifacts/evals/<run>/``.

The prompts deliberately include (a) in-domain Sheng, (b) Swahili-heavy,
(c) English-heavy, and (d) text with words that do not occur in the training
transcripts. If a checkpoint only handles (a) it has overfit to replaying the
training audio - the most common fine-tuning failure (upstream FAQ #169).

Export: package the adapter so it is portable - ``lora_config.json`` is rewritten
to reference ``openbmb/VoxCPM2`` instead of a personal Drive path.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import BASE_MODEL_ID, DEFAULT_RUN, Paths
from .progress import Progress
from .train import checkpoints, latest_checkpoint, resume_step

EVAL_TEXTS: list[dict[str, str]] = [
    {
        "id": "sheng_casual",
        "text": "Niaje msee, leo tuna chill tu hapa na mambo ya biz ndio zinaendelea fiti.",
    },
    {
        "id": "sheng_swahili",
        "text": "Wakurugenzi walikuwa wamekuja kufanya mkutano wa bajeti ya mwaka, halafu wakaenda zao.",
    },
    {
        "id": "sheng_english",
        "text": "Honestly speaking, hii kitu inafaa ifanyike mapema before the deadline ifike.",
    },
    {
        "id": "unseen_vocab",
        "text": "Taarifa ya hali ya hewa inasema kesho kutakuwa na joto kali sana mjini Nairobi.",
    },
    {
        "id": "long_form",
        "text": (
            "Sasa hivi tunaskia ati kuna scholarship mpya imetolewa na serikali kwa vijana wote "
            "wanaomaliza shule, kwa hivyo ni vyema kila mtu akaapply mapema na documents zote."
        ),
    },
]


@dataclass
class EvalOptions:
    run: str = DEFAULT_RUN
    steps: list[int] | None = None  # None -> latest checkpoint
    ab: bool = False  # also synthesize with LoRA disabled (base model)
    limit: int | None = None  # only the first N eval texts
    optimize: bool = False  # torch.compile; slower to start, faster per clip
    device: str = "cuda"


@dataclass
class ExportOptions:
    run: str = DEFAULT_RUN
    step: int | None = None  # None -> latest
    include_samples: bool = True


# --------------------------------------------------------------------- eval
def _pick_checkpoints(paths: Paths, run: str, steps: list[int] | None) -> list[tuple[int, Path]]:
    save_dir = paths.checkpoint_dir(run)
    available = dict(checkpoints(save_dir))
    if steps:
        picked = []
        for step in steps:
            if step not in available:
                raise SystemExit(f"step {step} not found under {save_dir}; have {sorted(available)}")
            picked.append((step, available[step]))
        return picked
    latest = latest_checkpoint(save_dir)
    if latest is None:
        raise SystemExit(f"no checkpoints under {save_dir} - run `sheng-voxcpm train --resume` first")
    step = resume_step(save_dir) or (max(available) if available else 0)
    return [(step, latest)]


def _lora_config(paths: Paths, info: dict[str, Any]):
    """Rebuild the upstream ``LoRAConfig`` dataclass from ``lora_config.json``.

    ``VoxCPM.from_pretrained`` expects the dataclass, not the raw dict that the
    trainer wrote into the checkpoint. VoxCPM1.x and VoxCPM2 have separate
    classes, picked here from the base model's ``config.json`` architecture.
    """
    cfg = info.get("lora_config")
    if not cfg:
        return None
    architecture = "voxcpm2"
    try:
        architecture = str(
            json.loads(paths.base_config_json.read_text(encoding="utf-8")).get("architecture", "voxcpm2")
        ).lower()
    except (OSError, json.JSONDecodeError):
        pass
    if architecture == "voxcpm2":
        from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2

        return LoRAConfigV2(**cfg)
    from voxcpm.model.voxcpm import LoRAConfig as LoRAConfigV1

    return LoRAConfigV1(**cfg)


def _load_model(paths: Paths, ckpt: Path, *, optimize: bool, device: str):
    import inspect

    import torch
    from voxcpm.core import VoxCPM

    info = {}
    cfg_file = ckpt / "lora_config.json"
    if cfg_file.exists():
        info = json.loads(cfg_file.read_text(encoding="utf-8"))

    kwargs: dict[str, Any] = {}
    # `device` was added to the loader in VoxCPM 2.0.3; older revisions pick the
    # device themselves (cuda when available), so only pass it when supported.
    if "device" in inspect.signature(VoxCPM.from_pretrained).parameters:
        kwargs["device"] = device
    elif device not in ("cuda", "auto"):
        raise SystemExit(f"this VoxCPM revision cannot select device={device!r}; use an updated checkout")

    model = VoxCPM.from_pretrained(
        hf_model_id=str(paths.model_dir),
        load_denoiser=False,
        optimize=optimize,
        lora_weights_path=str(ckpt),
        lora_config=_lora_config(paths, info),
        **kwargs,
    )
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available for evaluation")
    return model


def evaluate(paths: Paths, opts: EvalOptions, *, log=print) -> dict[str, Any]:
    import soundfile as sf

    paths.ensure()
    progress = Progress(paths)
    out_dir = paths.eval_dir(opts.run)
    texts = EVAL_TEXTS[: opts.limit] if opts.limit else EVAL_TEXTS
    picked = _pick_checkpoints(paths, opts.run, opts.steps)

    progress.event("eval", "running", note=f"steps={[s for s, _ in picked]}", run=opts.run)
    results: list[dict[str, Any]] = []
    sample_rate = None

    for step, ckpt in picked:
        log(f"[eval] loading {ckpt} (step {step})")
        model = _load_model(paths, ckpt, optimize=opts.optimize, device=opts.device)
        sample_rate = int(model.tts_model.sample_rate)
        step_dir = out_dir / f"step_{step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for item in texts:
            for variant in (["lora"] + (["base"] if opts.ab else [])):
                if variant == "base":
                    model.set_lora_enabled(False)
                wav_path = step_dir / f"{item['id']}__{variant}.wav"
                t0 = time.time()
                audio = model.generate(
                    text=item["text"],
                    cfg_value=2.0,
                    inference_timesteps=10,
                    max_len=1000,
                    normalize=False,
                    denoise=False,
                    retry_badcase=True,
                    seed=42,
                )
                gen_s = time.time() - t0
                sf.write(str(wav_path), audio, sample_rate)
                dur = len(audio) / sample_rate
                results.append(
                    {
                        "step": step,
                        "variant": variant,
                        "id": item["id"],
                        "text": item["text"],
                        "wav": str(wav_path),
                        "audio_sec": round(dur, 2),
                        "gen_sec": round(gen_s, 2),
                        "rtf": round(gen_s / dur, 3) if dur else None,
                    }
                )
                log(f"[eval]   {item['id']:<14} {variant:<4} {dur:5.2f}s audio in {gen_s:5.2f}s -> {wav_path.name}")
        model.set_lora_enabled(True)
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    summary = {
        "run": opts.run,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_rate": sample_rate,
        "steps": [s for s, _ in picked],
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = _render_eval_report(summary, out_dir)
    (out_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")

    progress.event("eval", "done", note=f"{len(results)} wavs -> {out_dir}", run=opts.run)
    log(f"[eval] report -> {out_dir / 'report.md'}")
    return summary


def _render_eval_report(summary: dict[str, Any], out_dir: Path) -> str:
    lines = [
        f"# Eval - run `{summary['run']}`",
        "",
        f"Base model: `{BASE_MODEL_ID}` | output sample rate: {summary['sample_rate']} Hz",
        f"Generated: {summary['generated']} | steps: {summary['steps']}",
        "",
        "Listen in this order: `sheng_casual` (in-domain), `unseen_vocab` and "
        "`long_form` (text adherence), then compare `*__lora.wav` vs `*__base.wav`.",
        "",
        "| step | variant | prompt | audio s | gen s | RTF | file |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary["results"]:
        rel = Path(r["wav"]).name
        lines.append(
            f"| {r['step']} | {r['variant']} | {r['id']} | {r['audio_sec']} | "
            f"{r['gen_sec']} | {r['rtf']} | `{rel}` |"
        )
    lines += [
        "",
        "## How to judge",
        "",
        "- **Accent/timbre**: should sound like the dataset speaker, including English words.",
        "- **Text adherence**: `unseen_vocab`/`long_form` must say the new words, not replay training audio.",
        "- **Stability**: no long silences, no repeated syllables; RTF should be < 1 on L4.",
        "- Pick the *earliest* checkpoint that sounds right; later ones usually overfit.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- export
def export(paths: Paths, opts: ExportOptions, *, log=print) -> Path:
    paths.ensure()
    progress = Progress(paths)
    save_dir = paths.checkpoint_dir(opts.run)
    if opts.step is not None:
        ckpt = save_dir / f"step_{opts.step:07d}"
        if not ckpt.is_dir():
            ckpt = save_dir / f"step_{opts.step}"
        if not ckpt.is_dir():
            raise SystemExit(f"checkpoint for step {opts.step} not found in {save_dir}")
        step = opts.step
    else:
        ckpt = latest_checkpoint(save_dir)
        if ckpt is None:
            raise SystemExit(f"no checkpoint in {save_dir} - train first")
        step = resume_step(save_dir)

    dest = paths.export_dir(opts.run) / f"step_{step:07d}"
    dest.mkdir(parents=True, exist_ok=True)

    weights = ckpt / "lora_weights.safetensors"
    if not weights.exists():
        alt = ckpt / "lora_weights.ckpt"
        if not alt.exists():
            raise SystemExit(f"no lora weights in {ckpt}")
        weights = alt
    shutil.copy2(weights, dest / weights.name)

    info = {}
    cfg_file = ckpt / "lora_config.json"
    if cfg_file.exists():
        info = json.loads(cfg_file.read_text(encoding="utf-8"))
    info["base_model"] = BASE_MODEL_ID  # portable: no personal Drive path
    (dest / "lora_config.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    eval_dir = paths.eval_dir(opts.run) / f"step_{step:07d}"
    if opts.include_samples and eval_dir.is_dir():
        samples = dest / "samples"
        samples.mkdir(exist_ok=True)
        for wav in sorted(eval_dir.glob("*.wav")):
            shutil.copy2(wav, samples / wav.name)

    lora_cfg = info.get("lora_config", {})
    (dest / "README.md").write_text(
        "\n".join(
            [
                f"# Sheng LoRA for VoxCPM2 - step {step}",
                "",
                f"- base model: `{info['base_model']}`",
                f"- LoRA: r={lora_cfg.get('r')} alpha={lora_cfg.get('alpha')} "
                f"lm={lora_cfg.get('enable_lm')} dit={lora_cfg.get('enable_dit')} proj={lora_cfg.get('enable_proj')}",
                f"- dataset: `lyle-mlengineer/kenyan-celebs` (Sheng: Swahili + English)",
                "",
                "## Load",
                "",
                "```python",
                "from voxcpm.core import VoxCPM",
                "",
                "model = VoxCPM.from_pretrained(",
                f'    hf_model_id="{info["base_model"]}",',
                "    load_denoiser=False,",
                f'    lora_weights_path="{dest.as_posix()}",',
                ")",
                'audio = model.generate(text="Niaje msee, leo tuna chill tu.", seed=42)',
                "# model.tts_model.sample_rate -> 48000",
                "```",
                "",
                "`lora_config.json` next to the weights carries r/alpha/target modules; "
                "`lora_weights_path` must point at the directory that contains it.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    progress.event("export", "done", note=str(dest), step=step)
    log(f"[export] packaged step {step} -> {dest}")
    return dest
