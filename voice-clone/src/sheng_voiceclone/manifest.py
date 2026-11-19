"""JSONL manifests - the one format every stage reads and writes.

A line looks like::

    {"audio": "data/wav16k/utt_0001.wav", "text": "Sasa bro, uko aje?",
     "speaker": "abel", "language": "sw", "seconds": 30.0}

``audio`` is resolved relative to the artifacts root when it is not absolute, so a
manifest stays valid when Drive is re-mounted at a different path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

REQUIRED = ("audio", "text")


class ManifestError(ValueError):
    """A manifest is unusable: missing file, missing key, or empty."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f"manifest not found: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{p}:{lineno}: not JSON ({exc})") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
    return p


def resolve_audio(row: dict[str, Any], root: Path | None = None) -> Path:
    p = Path(str(row["audio"]))
    return p if p.is_absolute() else (Path(root) / p if root else p)


def validate(
    rows: list[dict[str, Any]],
    *,
    root: Path | None = None,
    check_files: bool = True,
) -> list[str]:
    """Return a list of problems; empty means the manifest is usable."""
    problems: list[str] = []
    if not rows:
        return ["manifest has no rows"]
    for i, row in enumerate(rows, start=1):
        for key in REQUIRED:
            if not row.get(key):
                problems.append(f"row {i}: missing '{key}'")
        if not row.get("text") or not str(row["text"]).strip():
            problems.append(f"row {i}: empty text")
        if check_files and row.get("audio"):
            p = resolve_audio(row, root)
            if not p.is_file():
                problems.append(f"row {i}: audio not found: {p}")
    return problems


def load(
    path: str | Path,
    *,
    root: Path | None = None,
    check_files: bool = True,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    problems = validate(rows, root=root, check_files=check_files)
    if problems:
        head = "\n  ".join(problems[:10])
        more = "" if len(problems) <= 10 else f"\n  ... {len(problems) - 10} more"
        raise ManifestError(f"{path}: {len(problems)} problem(s)\n  {head}{more}")
    return rows


def rel_audio(path: Path, root: Path) -> str:
    """Store audio paths relative to the artifacts root (Drive portability)."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)
