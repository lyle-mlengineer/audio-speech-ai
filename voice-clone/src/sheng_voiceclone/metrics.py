"""Objective voice-clone scoring. Pure stdlib + the audio stats.

A cloned clip has two independent failure modes and one common one:

* **wrong words** - the model says something else (hallucination, truncation).
  Measured with CER/WER against the prompt text, via an ASR judge.
* **wrong voice** - the words are right but it is not the speaker. Measured with
  cosine similarity between speaker embeddings.
* **broken audio** - silence, clipping, runaway generation. Measured from the
  waveform alone, so this check needs no model and catches most real failures.

Every number here is *relative*: Sheng is code-switched, so a Whisper CER of 0.3
on the base model may be the ceiling, not a bug. What matters is base vs tuned.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

# ------------------------------------------------------------------- text prep
_APOSTROPHES = "'\u2019`"


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation/diacritics, collapse whitespace.

    Swahili orthography is phonemic and ASCII, so aggressive normalisation is
    safe here and stops the ASR judge being punished for punctuation only.
    """
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("\u2019", "'").replace("`", "'")
    out = []
    for ch in s:
        if ch.isalnum() or ch in _APOSTROPHES:
            out.append(ch)
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def words(s: str) -> list[str]:
    norm = normalize_text(s)
    return norm.split() if norm else []


def chars(s: str) -> list[str]:
    norm = normalize_text(s)
    return list(norm.replace(" ", ""))


# ------------------------------------------------------------------ edit costs
def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences (O(len(a)*len(b)))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,          # deletion
                cur[j - 1] + 1,       # insertion
                prev[j - 1] + (0 if ai == bj else 1),  # substitution
            )
        prev = cur
    return prev[-1]


def _rate(ref: Sequence[str], hyp: Sequence[str]) -> float | None:
    if not ref:
        return None
    return edit_distance(ref, hyp) / float(len(ref))


def wer(ref: str, hyp: str) -> float | None:
    """Word error rate; ``None`` when the reference has no words."""
    return _rate(words(ref), words(hyp))


def cer(ref: str, hyp: str) -> float | None:
    """Character error rate; ``None`` when the reference has no characters."""
    return _rate(chars(ref), chars(hyp))


# --------------------------------------------------------------- speaker match
def cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Cosine similarity of two speaker embeddings (``None`` if degenerate)."""
    import math

    if not a or not b or len(a) != len(b):
        return None
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (na * nb)


# ------------------------------------------------------------------- verdicts
SILENT_RMS = 1e-3        # -60 dBFS: effectively nothing came out
CLIP_FRAC = 0.01         # >1% of samples pinned at full scale
CER_BAD = 0.60           # more than 60% characters wrong: not the words
CER_GOOD = 0.35
SIM_BAD = 0.50           # cosine: a different speaker, not an accent shift
SIM_GOOD = 0.65


def judge(
    audio_stats: dict[str, float],
    *,
    expected_seconds: float | None = None,
    cer: float | None = None,
    similarity: float | None = None,
) -> tuple[str, str]:
    """Return ``(status, note)`` for one generated clip.

    ``status`` is one of ``ok`` / ``silent`` / ``clipping`` / ``runaway`` /
    ``truncated`` / ``wrong-words`` / ``wrong-voice``, checked in that order:
    a broken waveform makes the text/voice numbers meaningless.
    """
    s = audio_stats
    seconds = float(s.get("seconds", 0.0))
    if seconds <= 0.0 or float(s.get("rms", 0.0)) < SILENT_RMS:
        return "silent", f"rms={s.get('rms', 0.0):.5f} (< {SILENT_RMS}) - nothing synthesised"
    if float(s.get("clipped_frac", 0.0)) > CLIP_FRAC:
        return "clipping", f"{100 * float(s['clipped_frac']):.1f}% of samples at full scale - LR too high"
    if expected_seconds and seconds > 3.0 * expected_seconds:
        return "runaway", f"{seconds:.1f}s for ~{expected_seconds:.1f}s of text - EOS not learned"
    if expected_seconds and seconds < 0.35 * expected_seconds:
        return "truncated", f"{seconds:.1f}s for ~{expected_seconds:.1f}s of text - cut short"
    if cer is not None and cer > CER_BAD:
        return "wrong-words", f"cer={cer:.2f} - the text is not being said"
    if similarity is not None and similarity < SIM_BAD:
        return "wrong-voice", f"spk-sim={similarity:.2f} - different speaker, not just accent"
    return "ok", ""


def expected_seconds(text: str, chars_per_second: float = 13.0, floor: float = 1.0) -> float:
    """Rough spoken duration for a prompt, used to catch runaway/truncation.

    13 characters/s is Sheng delivered at a normal clip: Swahili syllables are
    open (CV), so this is close enough to flag a 3x overrun or a 3x cut.
    """
    n = len(chars(text))
    return max(floor, n / chars_per_second)


def summarise(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scored rows into the numbers a compare table shows."""
    rows = list(rows)
    cers = [r["cer"] for r in rows if r.get("cer") is not None]
    wers = [r["wer"] for r in rows if r.get("wer") is not None]
    sims = [r["similarity"] for r in rows if r.get("similarity") is not None]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
    return {
        "clips": len(rows),
        "ok": statuses.get("ok", 0),
        "statuses": statuses,
        "cer_mean": (sum(cers) / len(cers)) if cers else None,
        "wer_mean": (sum(wers) / len(wers)) if wers else None,
        "similarity_mean": (sum(sims) / len(sims)) if sims else None,
        "seconds_total": sum(float(r.get("seconds", 0.0)) for r in rows),
    }


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_scores_md(title: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """A markdown table of clip scores plus the aggregate."""
    lines = [
        f"# {title}",
        "",
        f"* clips: **{summary['clips']}** ({summary['ok']} ok)",
        f"* CER mean: **{_fmt(summary['cer_mean'])}** | WER mean: **{_fmt(summary['wer_mean'])}**"
        f" | speaker sim mean: **{_fmt(summary['similarity_mean'])}**",
    ]
    if summary["statuses"]:
        bits = ", ".join(f"{k}={v}" for k, v in sorted(summary["statuses"].items()))
        lines.append(f"* verdicts: {bits}")
    lines += ["", "| clip | sec | cer | wer | spk-sim | status | note |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            "| {clip} | {sec} | {cer} | {wer} | {sim} | {status} | {note} |".format(
                clip=r.get("clip", "?"),
                sec=_fmt(r.get("seconds"), 1),
                cer=_fmt(r.get("cer")),
                wer=_fmt(r.get("wer")),
                sim=_fmt(r.get("similarity")),
                status=r.get("status", "?"),
                note=(r.get("note") or "").replace("|", "/"),
            )
        )
    lines.append("")
    lines.append(
        "Reading this: **status** is the verdict (a broken waveform is checked first, "
        "then words, then voice). Numbers are only meaningful *relative* to the "
        "zero-shot baseline of the same prompts - run `compare`."
    )
    return "\n".join(lines)
