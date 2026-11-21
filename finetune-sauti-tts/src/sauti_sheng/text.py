"""Sheng-aware text helpers.

The dataset transcripts are verbatim, punctuated monologue. They are the ground
truth for what is said in each 30 s clip, so the data stage does **not** rewrite
them: no number expansion, no apostrophe stripping, no lowercasing. Anything
that changes the characters is a chance to misalign text and audio, and F5-TTS
is character-based - it will happily learn ``ng'ang'ana`` exactly as written.

The helpers here only do lossless clean-up (whitespace, curly quotes) plus the
two things the pipeline genuinely needs:

* a sentence-boundary estimate for the reference clip (we need a ~9 s prefix
  with matching text, and there are no word timestamps in the parquet);
* a character-coverage check against the Sauti vocab (an OOV character becomes
  index 0 = space, which silently corrupts training).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

#: Short, hand-written probes that mix Swahili and English the way the dataset
#: does. Keep them under ~110 characters: F5 conditions on the reference clip
#: and the total (reference + generation) should stay under ~30 s.
SHENG_PROBES: tuple[str, ...] = (
    "Niaje mazee, unafanya nini leo? Tunataka kupiga story kidogo na wewe.",
    "Hiyo maneno yako ni fiti sana, lakini lazima tuongee ukweli.",
    "Weka pesa kwa M-Pesa, alafu nitakutumia confirmation message.",
)

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "\u00a0": " "})
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Lossless clean-up: whitespace, curly quotes, control characters."""
    text = str(text).translate(_QUOTES)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation followed by whitespace.

    Deliberately simple. The transcripts are conversational and full of
    "Mmmh."/"Eeeh." fragments; a heavyweight sentence splitter would only
    invent boundaries that are not pauses in the audio.
    """
    text = clean_text(text)
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or [text]


def join_sentences(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip()).strip()


@dataclass(frozen=True)
class PrefixChoice:
    text: str
    fraction: float
    n_sentences: int
    total_sentences: int


def choose_ref_prefix(text: str, target_fraction: float) -> PrefixChoice:
    """Longest sentence-aligned prefix whose character share is closest to target.

    Speech rate is roughly stable within one speaker, so "characters spoken so
    far / total characters" is a usable estimate of "seconds spoken so far" when
    there are no timestamps. It is an estimate, not an alignment - the data
    report says so, and the notebook plays the clip so you can check it.
    """
    target_fraction = min(max(float(target_fraction), 0.05), 0.95)
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return PrefixChoice(clean_text(text), 1.0, 1, 1)

    total = max(1, len(clean_text(text)))
    target_chars = target_fraction * total
    best: tuple[float, int, str] | None = None
    for i in range(1, len(sentences)):
        prefix = join_sentences(sentences[:i])
        frac = len(prefix) / total
        # Do not let the search collapse onto a two-word first sentence.
        if len(prefix) < 20:
            continue
        score = abs(frac - target_fraction)
        if best is None or score < best[0]:
            best = (score, i, prefix)

    if best is None:
        prefix = clean_text(text)
        return PrefixChoice(prefix, 1.0, 1, len(sentences))
    _, i, prefix = best
    return PrefixChoice(prefix, len(prefix) / total, i, len(sentences))


def estimate_fraction(target_seconds: float, total_seconds: float) -> float:
    if total_seconds <= 0:
        return 0.5
    return min(max(target_seconds / total_seconds, 0.05), 0.95)


def truncate_for_eval(text: str, max_chars: int = 170) -> str:
    """Keep an evaluation prompt short enough for F5's ~30 s window."""
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    sentences = split_sentences(window)
    if len(sentences) > 1:
        return join_sentences(sentences[:-1])
    cut = window.rsplit(" ", 1)[0]
    return cut if len(cut) >= 20 else window


@dataclass(frozen=True)
class CharsetReport:
    unique_chars: int
    missing: dict[str, int]
    non_ascii: dict[str, int]
    digits: int
    apostrophes: int

    @property
    def clean(self) -> bool:
        return not self.missing


def charset_report(texts: list[str], vocab: list[str] | set[str]) -> CharsetReport:
    vocab_set = set(vocab)
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(text)
    missing = {ch: n for ch, n in counts.items() if ch not in vocab_set}
    non_ascii = {ch: n for ch, n in counts.items() if ord(ch) > 127}
    return CharsetReport(
        unique_chars=len(counts),
        missing=missing,
        non_ascii=non_ascii,
        digits=sum(n for ch, n in counts.items() if ch.isdigit()),
        apostrophes=counts.get("'", 0),
    )


def read_vocab(path) -> list[str]:
    """One character per line, exactly as F5's ``get_tokenizer`` reads it.

    The file ends with a newline, so the naive ``split("\\n")`` adds a phantom
    empty entry; F5 drops it by stripping the last character of every line.
    CRLF files are normalised so a stray ``\\r`` cannot become a token.
    """
    out: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line:
                out.append(line)
    return out


def f5_text_identity(text: str) -> bool | None:
    """Does F5's ``convert_char_to_pinyin`` leave this text unchanged?

    Training stores verbatim transcripts; inference passes them through
    ``convert_char_to_pinyin`` (the pinyin G2P for Chinese). For Latin-only
    Sheng the transform is identity - checked for every dataset row - so the two
    agree. This helper lets the data/train stages prove it instead of assuming.
    Returns None when f5-tts is not installed.
    """
    try:
        from f5_tts.model.utils import convert_char_to_pinyin
    except Exception:
        return None
    return "".join(convert_char_to_pinyin([text])[0]) == text


def chars_per_second(texts: list[str], durations: list[float]) -> float:
    chars = sum(len(t) for t in texts)
    secs = sum(d for d in durations if d > 0)
    return chars / secs if secs else 0.0
