"""Chatterbox T3 fine-tuning pipeline for Sheng (Swahili + English code-switching).

Chatterbox is `ve` (speaker embedding) + `t3` (text -> speech tokens) + `s3gen`
(speech tokens -> 24 kHz audio). Only **T3** is fine-tuned; the rest is frozen,
so a run changes *who* the model sounds like without touching audio quality.

Everything expensive the pipeline produces (audio, manifests, checkpoints, logs,
progress state) is written under an ``artifacts`` root so that on Colab it can
live on Google Drive and survive a runtime disconnect.

Quick start on Colab::

    %cd /content/audio-speech-ai/finetune-chatterbox
    !pip install -q -e .
    !sheng-chatterbox all --resume

After a disconnect, re-run the same two commands: finished stages are skipped and
training resumes from ``<artifacts>/checkpoints/<run>/latest``.
"""

from __future__ import annotations

__version__ = "0.1.0"
