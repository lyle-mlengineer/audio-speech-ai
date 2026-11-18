"""Shared fixtures. Tests import the package from ``src/`` (no install needed).

Also carries a 40-line YAML reader for the subset our renderer/templates emit, so
the config tests do not depend on PyYAML being built for the local Python.
"""

from __future__ import annotations

import json
import re
import sys
import wave
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sheng_chatterbox.paths import Paths  # noqa: E402 - needs sys.path above

_INT = re.compile(r"^-?\d+$")
_FLOAT = re.compile(r"^-?\d*\.\d+(?:[eE][-+]?\d+)?$")


def _strip_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _scalar(value: str):
    if value.startswith(('"', "'")):
        return json.loads(value) if value.startswith('"') else value[1:-1]
    if value == "true":
        return True
    if value == "false":
        return False
    if value in ("null", "~"):
        return None
    if _INT.match(value):
        return int(value)
    if _FLOAT.match(value):
        return float(value)
    return value


def parse_yaml(text: str) -> dict:
    entries: list[tuple[int, str, str]] = []
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        entries.append((indent, key.strip(), value.strip()))

    def block(pos: int, parent_indent: int):
        out: dict = {}
        while pos < len(entries) and entries[pos][0] > parent_indent:
            indent, key, value = entries[pos]
            if value == "":
                out[key], pos = block(pos + 1, indent)
            else:
                out[key] = _scalar(value)
                pos += 1
        return out, pos

    parsed, _ = block(0, -1)
    return parsed


def make_wav(path: Path, seconds: float = 3.0, sample_rate: int = 16_000) -> Path:
    """Tiny silent mono PCM16 WAV, written with stdlib only (no numpy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * frames)
    return path


@pytest.fixture()
def wav_factory():
    return make_wav


@pytest.fixture()
def mini_yaml():
    """Stand-in for ``yaml.safe_load`` on the subset we emit."""
    return parse_yaml


@pytest.fixture(autouse=True)
def _isolate_progress_md(monkeypatch):
    """Never let a test rewrite the repo's real progress.md.

    ``Progress.write_markdown`` mirrors the journal into the repo root; pointing
    that property at the (temporary) artifacts dir for every test keeps the
    checked-in progress.md untouched.
    """
    monkeypatch.setattr(Paths, "repo_progress_md", property(lambda self: self.progress_dir / "progress.md"))


@pytest.fixture(autouse=True)
def _isolate_artifacts_env(monkeypatch):
    """Tests must never resolve the developer's real artifacts root."""
    monkeypatch.delenv("SHENG_CHATTERBOX_ARTIFACTS", raising=False)
    monkeypatch.delenv("SHENG_ARTIFACTS", raising=False)


@pytest.fixture()
def paths(tmp_path):
    """Paths rooted in a temp dir (see _isolate_progress_md for the mirror)."""
    return Paths(tmp_path / "artifacts").ensure()
