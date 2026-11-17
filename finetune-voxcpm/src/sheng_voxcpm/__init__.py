"""VoxCPM2 LoRA fine-tuning pipeline for Sheng (Swahili + English code-switching).

Everything expensive the pipeline produces (audio, manifests, checkpoints, logs,
progress state) is written under an ``artifacts`` root so that on Colab it can
live on Google Drive and survive a runtime disconnect.

Quick start on Colab::

    %cd /content/finetune-voxcpm
    !pip install -q -e .
    !sheng-voxcpm all --resume

After a disconnect, re-run the same two commands: finished stages are skipped and
training resumes from ``<artifacts>/checkpoints/<run>/latest``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
