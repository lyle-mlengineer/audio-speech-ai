"""VoxCPM2 LoRA fine-tuning pipeline for Sheng (Swahili + English code-switching).

Everything the pipeline writes (audio, manifests, checkpoints, logs) goes to an
``artifacts`` root so that on Colab it can live on Google Drive and survive a
runtime disconnect.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
