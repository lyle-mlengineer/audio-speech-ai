"""The readable templates in ``configs/`` must match what the renderer writes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from kenya_voices.config import PROFILES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_template_matches_renderer(profile):
    path = ROOT / "configs" / f"voxcpm2_design_lora_{profile}.yaml"
    assert path.exists(), "run `python scripts/render_configs.py`"
    before = path.read_text(encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "render_configs.py")], check=True, capture_output=True)
    assert path.read_text(encoding="utf-8") == before
    body = before.split("\n", 3)[3]
    # Paths are POSIX in the template even when rendered on Windows.
    assert "/content/drive/MyDrive/kenya-voices" in body
    assert "sample_rate: 16000" in body
