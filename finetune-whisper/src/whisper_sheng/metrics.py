"""Metrics for Sheng ASR.

Beyond plain WER we report two things that matter for code-switched speech:

1. `mixed_wer` — WER restricted to utterances that actually contain English
   words. Sheng switches mid-sentence, and aggregate WER can hide the fact
   that a model transcribes the Swahili parts fine while mangling the
   English insertions (or vice versa).

2. `code_switch_rate_error` — how often the model reproduces the reference's
   language mix. Cheap proxy: the absolute difference between the fraction
   of English tokens in the reference vs the hypothesis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import jiwer

from .text_norm import normalize_text, tokens

# A pragmatic English detector. Sheng is Swahili-based, so we score a token
# as English if it looks like an English word that is NOT a common Swahili
# word. This is intentionally approximate: it only feeds secondary metrics.
_ENGLISH_PROBE = re.compile(r"^[a-z']+$")

# High-frequency Swahili/Sheng function words. Kept small on purpose --
# these are the tokens most likely to be mis-flagged as English.
_SW_COMMON = {
    "na", "ya", "wa", "kwa", "ni", "la", "za", "hii", "hiyo", "hio", "huyo",
    "wale", "hao", "yeye", "mimi", "wewe", "sisi", "nyinyi", "wao", "kama",
    "lakini", "basi", "ndio", "si", "siyo", "hapana", "sana", "tu", "pia",
    "kwani", "nini", "gani", "wapi", "lini", "vipi", "mbona", "kuna", "hakuna",
    "niko", "uko", "yuko", "tuko", "mko", "wako", "ana", "una", "nina", "tuna",
    "alikuwa", "nilikuwa", "bana", "mzee", "mazee", "jo", "fit", "fiti", "sasa",
    "hata", "bado", "tena", "manze", "kweli", "sawa", "poa", "noma", "nje",
    "ndani", "juu", "chini", "mbele", "nyuma", "leo", "kesho", "jana", "siku",
    "mtu", "watu", "kitu", "vitu", "mahali", "mambo", "stori", "bangi", "rai",
}

# English tokens we expect to see in Sheng. Used only as a sanity signal.
_EN_COMMON = {
    "the", "and", "is", "are", "was", "were", "i", "you", "he", "she", "we",
    "they", "it", "a", "an", "to", "of", "in", "on", "at", "for", "with",
    "but", "so", "like", "just", "really", "actually", "okay", "ok", "yeah",
    "yes", "no", "not", "my", "your", "his", "her", "our", "their", "this",
    "that", "these", "those", "what", "when", "where", "who", "why", "how",
    "job", "bossy", "skill", "unmatched", "cool", "flavoured", "alright",
    "something", "everything", "nothing", "know", "think", "want", "got",
}

_EN_SUFFIX_HINTS = ("tion", "sion", "ment", "ness", "able", "ible", "ing", "ally")


def _looks_english(tok: str) -> bool:
    """Rough per-token English detector for code-switch statistics."""
    if tok in _EN_COMMON:
        return True
    if tok in _SW_COMMON:
        return False
    if not _ENGLISH_PROBE.match(tok):
        return False
    if len(tok) <= 2:
        return False
    if any(tok.endswith(sfx) for sfx in _EN_SUFFIX_HINTS):
        return True
    # Default: single short-ish tokens are ambiguous; treat as Swahili
    # (the base language) to avoid inflating the English rate.
    return False


@dataclass
class EvalReport:
    wer: float
    cer: float
    mixed_wer: float
    code_switch_rate_error: float
    n_examples: int
    n_mixed_examples: int
    per_example: List[Dict[str, object]] = field(default_factory=list)

    def to_metrics_dict(self) -> Dict[str, float]:
        """Scalar view for MLflow."""
        return {
            "wer": self.wer,
            "cer": self.cer,
            "mixed_wer": self.mixed_wer,
            "code_switch_rate_error": self.code_switch_rate_error,
            "n_eval_examples": float(self.n_examples),
            "n_mixed_examples": float(self.n_mixed_examples),
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "mixed_wer": self.mixed_wer,
            "code_switch_rate_error": self.code_switch_rate_error,
            "n_examples": self.n_examples,
            "n_mixed_examples": self.n_mixed_examples,
            "per_example": self.per_example,
        }


def english_token_fraction(text: str) -> float:
    toks = tokens(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if _looks_english(t)) / len(toks)


def compute_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    lowercase: bool = True,
    strip_punctuation: bool = True,
    collapse_repeats: bool = True,
    max_repeat_run: int = 2,
) -> EvalReport:
    """Compute the full metric bundle over a list of ref/hyp pairs."""
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references ({len(references)}) and hypotheses ({len(hypotheses)}) "
            "must have the same length"
        )

    norm_kwargs = dict(
        lowercase=lowercase,
        strip_punctuation=strip_punctuation,
        collapse_repeats=collapse_repeats,
        max_repeat_run=max_repeat_run,
    )

    refs_n = [normalize_text(r, **norm_kwargs) for r in references]
    # An empty hypothesis makes jiwer raise in some versions; substitute a
    # single blank token so the error is counted rather than crashing a run.
    hyps_n = [normalize_text(h, **norm_kwargs) or "" for h in hypotheses]
    hyps_n = [h if h else "" for h in hyps_n]

    per_example: List[Dict[str, object]] = []
    mixed_refs: List[str] = []
    mixed_hyps: List[str] = []
    cs_errors: List[float] = []

    for i, (ref, hyp) in enumerate(zip(refs_n, hyps_n)):
        ref_toks = tokens(ref)
        hyp_toks = tokens(hyp)

        # jiwer needs at least one token; guard degenerate rows.
        ref_for_score = ref if ref_toks else "<empty>"
        hyp_for_score = hyp if hyp_toks else "<empty>"

        wer_i = jiwer.wer(ref_for_score, hyp_for_score)
        cer_i = jiwer.cer(ref_for_score, hyp_for_score)

        ref_en = english_token_fraction(ref)
        hyp_en = english_token_fraction(hyp)
        cs_err = abs(ref_en - hyp_en)

        is_mixed = ref_en > 0.0
        if is_mixed:
            mixed_refs.append(ref_for_score)
            mixed_hyps.append(hyp_for_score)
        cs_errors.append(cs_err)

        per_example.append(
            {
                "index": i,
                "reference": ref,
                "hypothesis": hyp,
                "wer": float(wer_i),
                "cer": float(cer_i),
                "reference_english_fraction": float(ref_en),
                "hypothesis_english_fraction": float(hyp_en),
                "is_mixed": bool(is_mixed),
            }
        )

    overall_wer = float(jiwer.wer(refs_n, hyps_n)) if refs_n else float("nan")
    overall_cer = float(jiwer.cer(refs_n, hyps_n)) if refs_n else float("nan")

    # Correlation between the two normalised lists is already handled by
    # jiwer's multi-reference path, but empty strings break it, so score
    # mixed subsets with the same guards as above.
    if mixed_refs:
        mixed_wer = float(
            jiwer.wer(
                [r if tokens(r) else "<empty>" for r in mixed_refs],
                [h if tokens(h) else "<empty>" for h in mixed_hyps],
            )
        )
    else:
        mixed_wer = float("nan")

    cs_rate_error = float(sum(cs_errors) / len(cs_errors)) if cs_errors else float("nan")

    return EvalReport(
        wer=overall_wer,
        cer=overall_cer,
        mixed_wer=mixed_wer,
        code_switch_rate_error=cs_rate_error,
        n_examples=len(refs_n),
        n_mixed_examples=len(mixed_refs),
        per_example=per_example,
    )
