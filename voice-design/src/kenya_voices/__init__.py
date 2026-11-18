"""Caption-conditioned VoxCPM2 LoRA: let users design Kenyan-sounding voices.

VoxCPM2 already supports *voice design*: a natural-language description in
parentheses in front of the text conditions the voice. Out of the box that space
is mostly English-accented; this package fine-tunes a LoRA so descriptions like
``"(A Kenyan woman in her thirties, mid-pitched, speaking Swahili)"`` actually
produce Kenyan voices, in Swahili and in the English words Kenyans mix in.

Everything expensive (16 kHz WAVs, caption manifests, base weights, checkpoints,
logs, progress state) is written under an ``artifacts`` root so that on Colab it
can live on Google Drive and survive a runtime disconnect.

Quick start on Colab::

    %cd /content/audio-speech-ai/voice-design
    !pip install -q -e .
    !kenya-voices all --resume --max-hours 8

After a disconnect, re-run the same two commands: finished stages are skipped and
training resumes from ``<artifacts>/checkpoints/<run>/latest``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
