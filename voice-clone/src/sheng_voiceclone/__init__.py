"""Sheng (Swahili + English) voice cloning: measure the clone, then fine-tune it.

This package is the *evaluation and orchestration* layer for voice cloning in
this repo. Training lives in the sibling projects (``finetune-chatterbox``,
``finetune-voxcpm``, ``finetune-qwentts``); what was missing was an honest way to
answer "is the cloned voice actually the speaker, and is it still saying the
words?" - which is what this does, before any GPU is spent.
"""

__version__ = "0.1.0"
