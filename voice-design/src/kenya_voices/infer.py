"""Voice-design evaluation + export.

Evaluation answers one question: **does the caption actually design the voice?**
Listening is still the final judge, but two things are measurable for free:

* median f0 of the generated audio (``audio_features.median_f0``) - a "male" caption
  must land lower than a "female" caption on the *same* text;
* paired caption adherence - fraction of male/female pairs where the ordering holds,
  for the LoRA and (with ``--ab``) the base model. If the LoRA does not beat the
  base model on this, the caption signal was not learned, whatever the loss says.

The prompt set is deliberately small and fixed so runs are comparable, and it
includes an unseen-speaker / unseen-vocabulary prompt: a checkpoint that only
reproduces training voices has overfit.

Export packages the adapter with a portable ``lora_config.json`` (``base_model``
rewritten to the Hub id) plus the design prompts that worked.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_features import median_f0
from .paths import BASE_MODEL_ID, DEFAULT_RUN, Paths
from .progress import Progress
from .train import checkpoints, latest_checkpoint, resume_step

# Same text, two captions -> a measurable ordering test.
PAIR_TEXT = "Habari yako, leo ni siku nzuri ya kufanya kazi na kukutana na marafiki."

EVAL_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "male_thirties",
        "pair": "gender",
        "expect": "low",
        "caption": "A Kenyan male voice in his thirties, low-pitched, measured, unhurried pace, speaking Swahili",
        "text": PAIR_TEXT,
    },
    {
        "id": "female_thirties",
        "pair": "gender",
        "expect": "high",
        "caption": "A Kenyan female voice in her thirties, bright, higher-pitched, steady conversational pace, speaking Swahili",
        "text": PAIR_TEXT,
    },
    {
        "id": "male_forties_nairobi",
        "pair": "age",
        "expect": "",
        "caption": "A Kenyan man from Nairobi in his forties, mid-pitched, steady conversational pace, speaking Swahili",
        "text": "Wakurugenzi walikuwa wamekuja kufanya mkutano wa bajeti ya mwaka, halafu wakaenda zao.",
    },
    {
        "id": "female_twenties_city",
        "pair": "age",
        "expect": "high",
        "caption": "A young Kenyan woman in her early twenties, bright, higher-pitched, brisk city pace, speaking Swahili with some English words",
        "text": "Honestly speaking, hii kitu inafaa ifanyike mapema before the deadline ifike.",
    },
    {
        "id": "sheng_male_twenties",
        "pair": "",
        "expect": "low",
        "caption": "A Kenyan male voice in his twenties, low-pitched, brisk city pace, Swahili mixed with English (Sheng-style code-switching)",
        "text": "Niaje msee, leo tuna chill tu hapa na mambo ya biz ndio zinaendelea fiti.",
    },
    {
        "id": "unseen_vocab",
        "pair": "",
        "expect": "high",
        "caption": "An authentic Kenyan female voice in her thirties, mid-pitched, Swahili speech",
        "text": "Taarifa ya hali ya hewa inasema kesho kutakuwa na joto kali sana mjini Nairobi.",
    },
    {
        "id": "long_form_narrator",
        "pair": "",
        "expect": "low",
        "caption": "A Kenyan male narrator in his fifties or older, low-pitched, measured, unhurried pace, natural Swahili pronunciation",
        "text": (
            "Sasa hivi tunaskia ati kuna scholarship mpya imetolewa na serikali kwa vijana wote "
            "wanaomaliza shule, kwa hivyo ni vyema kila mtu akaapply mapema na documents zote."
        ),
    },
]

LOW_F0_MAX = 155.0  # Hz - comfortably below the 165 Hz female boundary
HIGH_F0_MIN = 165.0


@dataclass
class EvalOptions:
    run: str = DEFAULT_RUN
    steps: list[int] | None = None  # None -> latest checkpoint
    ab: bool = False  # also synthesize with LoRA disabled (base model)
    limit: int | None = None
    optimize: bool = False
    device: str = "cuda"


@dataclass
class ExportOptions:
    run: str = DEFAULT_RUN
    step: int | None = None
    include_samples: bool = True


# --------------------------------------------------------------------- eval
def _pick_checkpoints(paths: Paths, run: str, steps: list[int] | None) -> list[tuple[int, Path]]:
    save_dir = paths.checkpoint_dir(run)
    available = dict(checkpoints(save_dir))
    if steps:
        picked = []
        for step in steps:
            if step not in available:
                raise RuntimeError(f"step {step} not found under {save_dir}; have {sorted(available)}")
            picked.append((step, available[step]))
        return picked
    latest = latest_checkpoint(save_dir)
    if latest is None:
        raise RuntimeError(f"no checkpoints under {save_dir} - run `kenya-voices train` first")
    step = resume_step(save_dir) or (max(available) if available else 0)
    return [(step, latest)]


def _lora_config(paths: Paths, info: dict[str, Any]):
    """Rebuild upstream's ``LoRAConfig`` dataclass from ``lora_config.json``."""
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
    if "device" in inspect.signature(VoxCPM.from_pretrained).parameters:
        kwargs["device"] = device
    elif device not in ("cuda", "auto"):
        raise RuntimeError(f"this VoxCPM revision cannot select device={device!r}; update the checkout")

    model = VoxCPM.from_pretrained(
        hf_model_id=str(paths.model_dir),
        load_denoiser=False,
        optimize=optimize,
        lora_weights_path=str(ckpt),
        lora_config=_lora_config(paths, info),
        **kwargs,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available for evaluation")
    return model


def _adherence(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired ordering test: male caption must sound lower than female caption."""
    by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for r in results:
        by_variant.setdefault(r["variant"], {})[r["id"]] = r

    checks: list[dict[str, Any]] = []
    for variant, items in by_variant.items():
        for pair_id in {i["pair"] for i in EVAL_PROMPTS if i["pair"]}:
            members = [p for p in EVAL_PROMPTS if p["pair"] == pair_id]
            low = next((items.get(p["id"]) for p in members if p["expect"] == "low"), None)
            high = next((items.get(p["id"]) for p in members if p["expect"] == "high"), None)
            if not low or not high or low["f0_hz"] is None or high["f0_hz"] is None:
                continue
            ok = low["f0_hz"] < high["f0_hz"]
            checks.append(
                {
                    "variant": variant,
                    "pair": pair_id,
                    "low_id": low["id"],
                    "high_id": high["id"],
                    "low_f0": low["f0_hz"],
                    "high_f0": high["f0_hz"],
                    "gap_hz": round(high["f0_hz"] - low["f0_hz"], 1),
                    "pass": ok,
                }
            )
    by_variant_score = {
        variant: round(sum(c["pass"] for c in checks if c["variant"] == variant) / max(1, len([c for c in checks if c["variant"] == variant])), 3)
        for variant in by_variant
    }
    return {"checks": checks, "score": by_variant_score}


def evaluate(paths: Paths, opts: EvalOptions, *, log=print) -> dict[str, Any]:
    paths.ensure()
    progress = Progress(paths)
    out_dir = paths.eval_dir(opts.run)
    prompts = EVAL_PROMPTS[: opts.limit] if opts.limit else EVAL_PROMPTS
    # Resolve checkpoints first: a missing run must fail with a useful message,
    # not with an ImportError from a dependency the user does not need yet.
    picked = _pick_checkpoints(paths, opts.run, opts.steps)

    import soundfile as sf

    progress.event("eval", "running", note=f"steps={[s for s, _ in picked]}", run=opts.run)
    results: list[dict[str, Any]] = []
    sample_rate = None

    for step, ckpt in picked:
        log(f"[eval] loading {ckpt} (step {step})")
        model = _load_model(paths, ckpt, optimize=opts.optimize, device=opts.device)
        sample_rate = int(model.tts_model.sample_rate)
        step_dir = out_dir / f"step_{step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for item in prompts:
            for variant in ["lora"] + (["base"] if opts.ab else []):
                if variant == "base":
                    model.set_lora_enabled(False)
                wav_path = step_dir / f"{item['id']}__{variant}.wav"
                t0 = time.time()
                audio = model.generate(
                    text=f"({item['caption']}){item['text']}",
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
                        "pair": item["pair"],
                        "expect": item["expect"],
                        "caption": item["caption"],
                        "text": item["text"],
                        "wav": str(wav_path),
                        "audio_sec": round(dur, 2),
                        "gen_sec": round(gen_s, 2),
                        "rtf": round(gen_s / dur, 3) if dur else None,
                        "f0_hz": median_f0(audio, sample_rate),
                    }
                )
                log(
                    f"[eval]   {item['id']:<20} {variant:<4} {dur:5.2f}s in {gen_s:5.2f}s "
                    f"f0={results[-1]['f0_hz']} -> {wav_path.name}"
                )
        model.set_lora_enabled(True)
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    adherence = _adherence(results)
    summary = {
        "run": opts.run,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_rate": sample_rate,
        "steps": [s for s, _ in picked],
        "adherence": adherence,
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(_render_eval_report(summary, out_dir), encoding="utf-8", newline="\n")

    best = None
    lora_score = adherence["score"].get("lora", 0.0)
    if lora_score >= (adherence["score"].get("base", 0.0) if opts.ab else 0.0):
        best = max(picked, key=lambda p: p[0])[0]
    progress.event(
        "eval",
        "done",
        note=f"{len(results)} wavs | caption adherence lora={lora_score} -> {out_dir}",
        run=opts.run,
    )
    if best is not None:
        progress.set_state(best_checkpoint=str(paths.checkpoint_dir(opts.run) / f"step_{best:07d}"))
    log(f"[eval] report -> {out_dir / 'report.md'}")
    return summary


def _render_eval_report(summary: dict[str, Any], out_dir: Path) -> str:
    adherence = summary.get("adherence", {})
    lines = [
        f"# Voice-design eval - run `{summary['run']}`",
        "",
        f"Base model: `{BASE_MODEL_ID}` | output sample rate: {summary['sample_rate']} Hz",
        f"Generated: {summary['generated']} | steps: {summary['steps']}",
        "",
        "## Caption adherence (male caption must sound lower than female caption)",
        "",
    ]
    for variant, score in adherence.get("score", {}).items():
        lines.append(f"- `{variant}`: **{score:.0%}** of gender pairs ordered correctly")
    if adherence.get("checks"):
        lines += ["", "| variant | pair | low caption | high caption | f0 low | f0 high | gap | ok |", "|---|---|---|---|---|---|---|---|"]
        for c in adherence["checks"]:
            lines.append(
                f"| {c['variant']} | {c['pair']} | {c['low_id']} | {c['high_id']} | "
                f"{c['low_f0']} Hz | {c['high_f0']} Hz | {c['gap_hz']} Hz | {'yes' if c['pass'] else 'NO'} |"
            )

    lines += [
        "",
        "## Clips",
        "",
        "| step | variant | prompt | caption | audio s | gen s | RTF | f0 Hz | file |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["results"]:
        caption = r["caption"].replace("|", "/")
        lines.append(
            f"| {r['step']} | {r['variant']} | {r['id']} | {caption} | {r['audio_sec']} | "
            f"{r['gen_sec']} | {r['rtf']} | {r['f0_hz']} | `{Path(r['wav']).name}` |"
        )
    lines += [
        "",
        "## How to judge",
        "",
        "- **Kenyan-ness**: the target accent should be present in Swahili *and* in English words.",
        "- **Caption adherence**: `male_thirties` must sound like a man, `female_thirties` like a woman;",
        "  the f0 table above is a proxy - trust your ears, not the Hz values.",
        "- **Text adherence**: `unseen_vocab` / `long_form_narrator` must say the new words, not replay training audio.",
        "- **No caption regression**: with `--ab`, captions that work without the LoRA should still work.",
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
            raise RuntimeError(f"checkpoint for step {opts.step} not found in {save_dir}")
        step = opts.step
    else:
        ckpt = latest_checkpoint(save_dir)
        if ckpt is None:
            raise RuntimeError(f"no checkpoint in {save_dir} - train first")
        step = resume_step(save_dir)

    dest = paths.export_dir(opts.run) / f"step_{step:07d}"
    dest.mkdir(parents=True, exist_ok=True)

    weights = ckpt / "lora_weights.safetensors"
    if not weights.exists():
        alt = ckpt / "lora_weights.ckpt"
        if not alt.exists():
            raise RuntimeError(f"no lora weights in {ckpt}")
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

    # The captions that the dataset actually produced: a ready-made design palette.
    palette: list[str] = []
    try:
        profiles = json.loads(paths.speaker_profiles.read_text(encoding="utf-8"))
        palette = [p["caption"] for p in profiles.values() if p.get("caption")][:12]
    except (OSError, json.JSONDecodeError):
        pass
    (dest / "design_prompts.json").write_text(
        json.dumps({"captions": palette, "texts": [p["text"] for p in EVAL_PROMPTS[:5]]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lora_cfg = info.get("lora_config", {})
    (dest / "README.md").write_text(
        "\n".join(
            [
                f"# Kenyan voice-design LoRA for VoxCPM2 - step {step}",
                "",
                f"- base model: `{info['base_model']}`",
                f"- LoRA: r={lora_cfg.get('r')} alpha={lora_cfg.get('alpha')} "
                f"lm={lora_cfg.get('enable_lm')} dit={lora_cfg.get('enable_dit')} proj={lora_cfg.get('enable_proj')}",
                "- data: `badrex/swahili-speech-400hr` (Kenyan Swahili, speaker-level captions)",
                "",
                "## Load and design a voice",
                "",
                "```python",
                "from voxcpm.core import VoxCPM",
                "",
                "model = VoxCPM.from_pretrained(",
                f'    hf_model_id="{info["base_model"]}",',
                "    load_denoiser=False,",
                f'    lora_weights_path="{dest.as_posix()}",',
                ")",
                'audio = model.generate(',
                '    text="(A Kenyan female voice in her thirties, mid-pitched, speaking Swahili)Habari yako?",',
                "    seed=42,",
                ")",
                "# model.tts_model.sample_rate -> 48000",
                "```",
                "",
                "`lora_config.json` next to the weights carries r/alpha/target modules, so",
                "`lora_weights_path` must point at the directory that contains it.",
                "`samples/` holds the eval clips and `design_prompts.json` the captions the",
                "adapter was trained on - good starting points for users.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    progress.event("export", "done", note=str(dest), step=step)
    log(f"[export] packaged step {step} -> {dest}")
    return dest


# -------------------------------------------------------------------- design
@dataclass
class DesignOptions:
    caption: str
    text: str
    out: Path | None = None  # None -> exports/design/<slug>.wav
    lora: Path | None = None  # lora_weights_path dir; None -> base model only
    run: str = DEFAULT_RUN
    seed: int = 42
    cfg_value: float = 2.0
    timesteps: int = 10
    device: str = "cuda"


def design(paths: Paths, opts: DesignOptions, *, log=print) -> Path:
    """Synthesize one clip from a description - the user-facing API of this repo."""
    import soundfile as sf
    from voxcpm.core import VoxCPM

    if not opts.caption.strip():
        raise RuntimeError("--caption is required: it is the voice description")

    kwargs: dict[str, Any] = {}
    if opts.lora is not None:
        kwargs["lora_weights_path"] = str(opts.lora)
        cfg_file = Path(opts.lora) / "lora_config.json"
        if cfg_file.exists():
            kwargs["lora_config"] = _lora_config(paths, json.loads(cfg_file.read_text(encoding="utf-8")))

    model = VoxCPM.from_pretrained(hf_model_id=str(paths.model_dir), load_denoiser=False, **kwargs)
    log(f"[design] caption: {opts.caption}")
    audio = model.generate(
        text=f"({opts.caption.rstrip('.')}){opts.text}",
        cfg_value=opts.cfg_value,
        inference_timesteps=opts.timesteps,
        max_len=1000,
        normalize=False,
        denoise=False,
        retry_badcase=True,
        seed=opts.seed,
    )
    sample_rate = int(model.tts_model.sample_rate)
    if opts.out is None:
        slug = "-".join(opts.caption.lower().split()[:6]).strip("-").replace("/", "-") or "design"
        out_dir = paths.export_root / opts.run / "design"
        out_dir.mkdir(parents=True, exist_ok=True)
        opts.out = out_dir / f"{slug}.wav"
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(opts.out), audio, sample_rate)
    log(f"[design] {len(audio) / sample_rate:.2f}s @ {sample_rate} Hz -> {opts.out}")
    return opts.out
