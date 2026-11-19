"""Synthesis + scoring: the part that actually answers "is this the same voice?".

Three lazy backends, each imported only when used, so ``status``/``data`` run on a
bare machine:

* **synth** - Chatterbox Multilingual (``sw`` is a supported language), zero-shot
  from the reference clip or with a fine-tuned T3 checkpoint swapped in;
* **asr** - any Whisper checkpoint via ``transformers``, used as the *text* judge
  (CER/WER against the prompt);
* **spk** - ECAPA speaker embeddings via ``speechbrain``, used as the *voice*
  judge (cosine against the reference).

Nothing here trains. Training is the sibling projects' job (``finetune-chatterbox``
does the T3 fine-tune these scores are meant to evaluate).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import metrics, prompts
from .audio import read_16k, read_wav, stats
from .paths import LANGUAGE_ID, Paths

DEFAULT_ASR = "openai/whisper-small"
DEFAULT_SPK = "speechbrain/spkrec-ecapa-voxceleb"


# --------------------------------------------------------------------- plans
def plan(prompt_list: list[dict[str, str]], out_dir: Path, *, tag: str) -> list[dict[str, Any]]:
    """Turn prompts into the rows everything downstream uses (one wav per prompt)."""
    rows: list[dict[str, Any]] = []
    for p in prompt_list:
        rows.append(
            {
                "clip": p["id"],
                "bucket": p.get("bucket", ""),
                "text": p["text"],
                "wav": str(Path(out_dir) / f"{p['id']}.wav"),
                "tag": tag,
                "expected_seconds": round(metrics.expected_seconds(p["text"]), 2),
            }
        )
    return rows


def write_plan(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    out = Path(out_dir) / "plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------- synth
def _torch():
    import torch  # noqa: F401

    return torch


def load_engine(device: str = "cuda", *, tuned_t3: Path | None = None) -> Any:
    """Load Chatterbox Multilingual, optionally with a fine-tuned T3 state dict."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    engine = ChatterboxMultilingualTTS.from_pretrained(device=device)
    if tuned_t3 is not None:
        from safetensors.torch import load_file

        sd = load_file(str(tuned_t3))
        missing, unexpected = engine.t3.load_state_dict(sd, strict=False)
        print(f"  tuned T3: {len(missing)} missing / {len(unexpected)} unexpected keys")
        if len(missing) > 0.1 * len(sd):
            raise RuntimeError(
                f"{tuned_t3} does not fit this T3 (vocab/layout changed?): "
                f"{len(missing)} of {len(engine.t3.state_dict())} keys missing"
            )
    engine.eval()
    return engine


def synthesize(
    rows: list[dict[str, Any]],
    *,
    ref_audio: Path,
    language: str = LANGUAGE_ID,
    device: str = "cuda",
    tuned_t3: Path | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    seed: int = 1337,
    only: list[str] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Generate one WAV per row. Existing WAVs are reused unless ``force``."""
    import numpy as np
    import soundfile as sf

    todo = [r for r in rows if (only is None or r["clip"] in only)]
    todo = [r for r in todo if force or not Path(r["wav"]).is_file()]
    if not todo:
        print(f"  synth: nothing to do ({len(rows)} clips already on disk)")
        return rows

    torch = _torch()
    torch.manual_seed(seed)
    engine = load_engine(device, tuned_t3=tuned_t3)
    print(f"  synth: {len(todo)} clip(s) from ref={Path(ref_audio).name} (language={language})")

    for r in todo:
        out = Path(r["wav"])
        out.parent.mkdir(parents=True, exist_ok=True)
        wav = engine.generate(
            r["text"],
            language_id=language,
            audio_prompt_path=str(ref_audio),
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        x = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        if x.ndim == 2:
            x = x.mean(axis=0) if x.shape[0] <= 2 else x.mean(axis=1)
        sr = int(getattr(engine, "sr", 24000) or 24000)
        sf.write(str(out), np.clip(x.reshape(-1), -1.0, 1.0), sr, subtype="PCM_16")
        print(f"    {r['clip']:<12} -> {out.name} ({x.size / sr:.1f}s)")
    return rows


# ----------------------------------------------------------------------- ASR
def asr_judge(model_id: str = DEFAULT_ASR, *, device: str = "cuda") -> Any:
    """A callable ``wav_path -> text``. Whisper-small is ~1 GB and good enough."""
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=0 if device == "cuda" else -1,
        chunk_length_s=30,
    )

    def transcribe(path: str | Path) -> str:
        return str(pipe(str(path), generate_kwargs={"language": "sw"})["text"]).strip()

    return transcribe


# ---------------------------------------------------------------- speaker sim
def spk_judge(model_id: str = DEFAULT_SPK, *, device: str = "cuda") -> Any:
    """A callable ``wav_path -> embedding`` (list[float]) for cosine comparison."""
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source=model_id,
        savedir=str(Path.home() / ".cache" / "spkrec-ecapa"),
        run_opts={"device": device},
    )

    def embed(path: str | Path) -> list[float]:
        x, sr = read_wav(path)
        if sr != 16000:
            x = read_16k(path)
            sr = 16000
        import torch

        with torch.no_grad():
            e = enc.encode_batch(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        return [float(v) for v in e.squeeze().reshape(-1).tolist()]

    return embed


# -------------------------------------------------------------------- scoring
def score_rows(
    rows: list[dict[str, Any]],
    *,
    ref_audio: Path,
    transcribe: Any | None = None,
    embed: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach stats, CER/WER and speaker similarity to each generated clip."""
    ref_emb = None
    if embed is not None:
        ref_emb = embed(ref_audio)

    scored: list[dict[str, Any]] = []
    for r in rows:
        wav = Path(r["wav"])
        row = dict(r)
        if not wav.is_file():
            row.update(status="missing", note="wav not generated", seconds=0.0)
            scored.append(row)
            continue
        x, sr = read_wav(wav)
        st = stats(x, sr)
        hyp = transcribe(wav) if transcribe is not None else None
        sim = None
        if ref_emb is not None:
            e = embed(wav)
            sim = metrics.cosine(ref_emb, e)
        cer = metrics.cer(r["text"], hyp) if hyp else None
        wer = metrics.wer(r["text"], hyp) if hyp else None
        status, note = metrics.judge(
            st, expected_seconds=r.get("expected_seconds"), cer=cer, similarity=sim
        )
        row.update(
            {
                "seconds": round(st["seconds"], 2),
                "rms": round(st["rms"], 5),
                "peak": round(st["peak"], 4),
                "clipped_frac": round(st["clipped_frac"], 4),
                "transcript": hyp,
                "cer": None if cer is None else round(cer, 4),
                "wer": None if wer is None else round(wer, 4),
                "similarity": None if sim is None else round(sim, 4),
                "status": status,
                "note": note,
            }
        )
        scored.append(row)
    return scored, metrics.summarise(scored)


def write_scores(
    paths: Paths, name: str, rows: list[dict[str, Any]], summary: dict[str, Any], *, title: str
) -> tuple[Path, Path]:
    """Persist ``scores/<name>.json`` + ``.md`` (what `compare` reads back)."""
    paths.scores.mkdir(parents=True, exist_ok=True)
    js = paths.scores / f"{name}.json"
    md = paths.scores / f"{name}.md"
    js.write_text(json.dumps({"name": name, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    md.write_text(metrics.render_scores_md(title, rows, summary), encoding="utf-8")
    return js, md


def read_scores(paths: Paths, name: str) -> dict[str, Any]:
    p = paths.scores / f"{name}.json"
    if not p.is_file():
        raise FileNotFoundError(f"no scores for '{name}' at {p} - run `baseline`/`eval` first")
    return json.loads(p.read_text(encoding="utf-8"))


# -------------------------------------------------------------------- compare
def compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Verdict for "did fine-tuning actually beat the zero-shot baseline?"."""
    sa, sb = a["summary"], b["summary"]

    def delta(key: str) -> float | None:
        va, vb = sa.get(key), sb.get(key)
        return None if va is None or vb is None else float(vb) - float(va)

    lines = [
        "| metric | baseline | tuned | delta | better |",
        "|---|---|---|---|---|",
    ]
    cer_d = delta("cer_mean")
    wer_d = delta("wer_mean")
    sim_d = delta("similarity_mean")
    lines.append(
        f"| ok clips | {sa['ok']}/{sa['clips']} | {sb['ok']}/{sb['clips']} | "
        f"{sb['ok'] - sa['ok']:+d} | higher |"
    )
    lines.append(
        f"| CER mean | {sa['cer_mean'] if sa['cer_mean'] is None else round(sa['cer_mean'], 3)} | "
        f"{sb['cer_mean'] if sb['cer_mean'] is None else round(sb['cer_mean'], 3)} | "
        f"{'-' if cer_d is None else f'{cer_d:+.3f}'} | **lower** |"
    )
    lines.append(
        f"| WER mean | {sa['wer_mean'] if sa['wer_mean'] is None else round(sa['wer_mean'], 3)} | "
        f"{sb['wer_mean'] if sb['wer_mean'] is None else round(sb['wer_mean'], 3)} | "
        f"{'-' if wer_d is None else f'{wer_d:+.3f}'} | **lower** |"
    )
    lines.append(
        f"| speaker sim | {sa['similarity_mean'] if sa['similarity_mean'] is None else round(sa['similarity_mean'], 3)} | "
        f"{sb['similarity_mean'] if sb['similarity_mean'] is None else round(sb['similarity_mean'], 3)} | "
        f"{'-' if sim_d is None else f'{sim_d:+.3f}'} | **higher** |"
    )
    lines.append("")

    win_sim = sim_d is not None and sim_d > 0.02
    keep_words = (cer_d is None or cer_d < 0.05) and (wer_d is None or wer_d < 0.05)
    if sim_d is None and cer_d is None:
        verdict = "NOT ENOUGH DATA - score both runs with ASR and/or a speaker embedder."
    elif win_sim and keep_words:
        verdict = "KEEP: closer to the speaker without losing the words."
    elif win_sim:
        verdict = "MIXED: closer voice but the words got worse - lower the LR or stop earlier."
    elif sa["ok"] > sb["ok"]:
        verdict = "REJECT: fine-tuning broke more clips than it fixed."
    else:
        verdict = "NO GAIN: the zero-shot clone is already as good - ship the base model."
    lines.append(f"**Verdict: {verdict}**")
    return "\n".join(lines)


def fingerprint(*parts: Any) -> str:
    """Short stable id for an eval config (used in file names)."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:8]


def eval_prompts(only: str | None) -> list[dict[str, str]]:
    return prompts.select(only)
