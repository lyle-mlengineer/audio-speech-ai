"""The Sheng evaluation prompts.

Two jobs: measure intelligibility of the code-switch, and measure whether the
voice is still the speaker after fine-tuning.

Rules that keep the numbers comparable across runs:

* never edit a prompt once a baseline has been scored against it - the CER of
  tuned vs base is only a difference if the text is identical;
* keep them in the dataset's register (Swahili matrix + English content words),
  because a 21-minute fine-tune teaches delivery and timbre, not new vocabulary;
* three buckets, so a failure is attributable: ``sheng`` (code-switched),
  ``swahili`` (clean Swahili), ``english`` (the model's second language).
"""

from __future__ import annotations

from typing import Any

PROMPTS: list[dict[str, str]] = [
    {
        "id": "sheng_01",
        "bucket": "sheng",
        "text": "Sasa bro, uko aje? Leo niko na plans za kwenda tao kutafuta msee.",
    },
    {
        "id": "sheng_02",
        "bucket": "sheng",
        "text": "Alikuwa ananiambia ati anataka kubuy gari mpya next month, but akanunua bike.",
    },
    {
        "id": "sheng_03",
        "bucket": "sheng",
        "text": "Tunapiga sherehe kesho, so make sure unakam na drinks zako mapema.",
    },
    {
        "id": "sheng_04",
        "bucket": "sheng",
        "text": "Msee wangu alifika late na excuse ya traffic jam ya CBD, akasema gari ilikuwa na shida.",
    },
    {
        "id": "swahili_01",
        "bucket": "swahili",
        "text": "Ningependa kukushukuru kwa msaada wako katika kazi hii muhimu.",
    },
    {
        "id": "swahili_02",
        "bucket": "swahili",
        "text": "Watoto wanacheza uwanjani huku wazazi wao wakizungumza kwa furaha.",
    },
    {
        "id": "english_01",
        "bucket": "english",
        "text": "Let us meet at the studio tomorrow and finish the record before noon.",
    },
]


def by_id(prompt_id: str) -> dict[str, str]:
    for p in PROMPTS:
        if p["id"] == prompt_id:
            return p
    raise KeyError(f"unknown prompt id: {prompt_id} (have: {', '.join(p['id'] for p in PROMPTS)})")


def select(only: str | None = None) -> list[dict[str, str]]:
    """All prompts, a comma-separated id list, or a whole bucket."""
    if not only:
        return list(PROMPTS)
    keys = [k.strip() for k in only.split(",") if k.strip()]
    buckets = {p["bucket"] for p in PROMPTS}
    chosen: list[dict[str, str]] = []
    for k in keys:
        if k in buckets:
            chosen += [p for p in PROMPTS if p["bucket"] == k]
        else:
            chosen.append(by_id(k))
    seen: set[str] = set()
    out = []
    for p in chosen:
        if p["id"] not in seen:
            seen.add(p["id"])
            out.append(p)
    return out


def as_rows(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prompt dicts -> the row shape the scorer consumes."""
    return [{"clip": p["id"], "text": p["text"], "bucket": p.get("bucket", "")} for p in prompts]
