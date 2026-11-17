"""Sheng-aware text normalisation.

Why this file exists
--------------------
Sheng has no standard orthography. The same word appears as "nomaaa",
"noma", "Nomaa", "naah" depending on how the transcriber heard the slang, and
the dataset contains Kesha-style elongations plus heavy English code-switching
("lakini mtaniruhusu tu niseme" next to "I was going into the story").

If we score raw strings, WER measures spelling inconsistency rather than
recognition quality. So both the reference and the prediction go through the
exact same normalisation before we compute any metric.

This module is used in two places, and it must stay identical in both:
  1. Building training labels (so the model learns the normalised form).
  2. Scoring predictions against references.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

# Characters we treat as word separators. Swahili apostrophes are kept
# because they carry meaning ("wa-na", "si-") and foreshortened slang uses
# them, but we normalise their unicode variant first.
_APOSTROPHES = {"\u2019", "\u2018", "\u02bc", "\u201b", "`"}

# Punctuation removed when strip_punctuation=True. We keep apostrophes and
# hyphens out of this set deliberately (see above).
_PUNCT_RE = re.compile(r"[^\w\s'\-]|_", flags=re.UNICODE)

_WHITESPACE_RE = re.compile(r"\s+")

# Repeated-character collapse: any run of the same char longer than N.
_REPEAT_RE = re.compile(r"(.)\1{1,}", flags=re.UNICODE)


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_punctuation: bool = True,
    collapse_repeats: bool = True,
    max_repeat_run: int = 2,
) -> str:
    """Normalise one transcript.

    Order matters: unicode-normalise, unify apostrophes, strip punctuation,
    then collapse repeats, then squash whitespace.
    """
    if text is None:
        return ""

    text = str(text)
    # NFKC folds fullwidth/compat forms so tokenisation is stable.
    text = unicodedata.normalize("NFKC", text)

    for ap in _APOSTROPHES:
        text = text.replace(ap, "'")

    if strip_punctuation:
        text = _PUNCT_RE.sub(" ", text)

    if lowercase:
        text = text.lower()

    if collapse_repeats:
        # caps repeats to `max_repeat_run` occurrences of the same char,
        # e.g. with max_repeat_run=2: "nomaaa" -> "nomaa"
        text = _REPEAT_RE.sub(lambda m: m.group(1) * max_repeat_run, text)

    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_batch(texts: Iterable[str], **kwargs) -> List[str]:
    return [normalize_text(t, **kwargs) for t in texts]


def tokens(text: str) -> List[str]:
    """Whitespace tokens of an already-normalised string."""
    return [t for t in text.split(" ") if t]
