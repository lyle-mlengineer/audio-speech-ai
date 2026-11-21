"""Build notebooks/colab_finetune_sauti_sheng.ipynb (kept as a script so the
notebook is reviewable in diffs and can be regenerated)."""

import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


md(
    """
# Fine-tune Sauti TTS to speak Sheng

[`msingiai/sauti-tts`](https://huggingface.co/msingiai/sauti-tts) (F5-TTS v1 Base, Swahili) fine-tuned
on [`lyle-mlengineer/kenyan-celebs`](https://huggingface.co/datasets/lyle-mlengineer/kenyan-celebs)
— 42 clips, 21 min, one speaker.

**Before you start:** `Runtime → Change runtime type → L4 GPU` (A100 if you have Pro; T4 only with
`--allow-small-gpu` and expect it to be slow).

Run the cells top to bottom. Stages are idempotent, so after a disconnect you re-run from the top and
it picks up where it stopped.
"""
)

md("## 1. GPU check")

code(
    """
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"""
)

code(
    """
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> L4 GPU"
props = torch.cuda.get_device_properties(0)
print(f"{props.name} | {props.total_memory / 1024**3:.1f} GB | capability {torch.cuda.get_device_capability(0)}")
"""
)

md(
    """
## 2. Mount Google Drive

Everything expensive (checkpoints, exports, progress state) is written to
`/content/drive/MyDrive/sauti-sheng`. The CLI detects the mount automatically.

Without Drive, artifacts land on the VM's local disk and die with the runtime — fine for a smoke test,
not for a real run.
"""
)

code(
    """
from google.colab import drive
drive.mount("/content/drive")
"""
)

md("## 3. Clone the repo")

code(
    """
import os
from pathlib import Path

REPO = "https://github.com/lyle-mlengineer/audio-speech-ai.git"
ROOT = Path("/content/audio-speech-ai")

if not (ROOT / ".git").is_dir():
    !git clone --depth 1 {REPO} {ROOT}
%cd {ROOT}
!git pull --ff-only
%cd {ROOT / "finetune-sauti-tts"}
print("cwd:", os.getcwd())
"""
)

md(
    """
## 4. Install

`f5-tts` is pinned to `1.1.22`: this recipe depends on `Trainer`, `CustomDataset` and the v1 Base DiT
layout. If pip changes torch and CUDA stops working, restart the runtime and continue from cell 5.
"""
)

code(
    """
%pip install -q -e .
%pip install -q f5-tts==1.1.22
"""
)

md(
    """
## 5. Stage 1 — environment check

Reports torch / F5 / vocos / datasets / ffmpeg / GPU / VRAM / free disk and picks a training profile
automatically (`l4`, `a100`, or `t4`).
"""
)

code(
    """
!sauti-sheng setup
!sauti-sheng status
"""
)

md(
    """
## 6. Stage 2 — data

Downloads the parquet and Sauti's own 2545-entry `vocab.txt`, resamples 22.05 kHz → 24 kHz
(F5 is 24 kHz only), peak-normalises, writes 42 WAVs plus the F5 `raw.arrow`, and builds a ~9 s
reference clip cut at a real pause in a **validation** clip (so training never hears it).

Prints the characters missing from the vocab and whether F5's `convert_char_to_pinyin` would alter any
transcript. Both should be clean for Sheng.
"""
)

code(
    """
!sauti-sheng data
"""
)

code(
    """
from pathlib import Path
from IPython.display import Markdown, Audio, display

ARTIFACTS = Path("/content/drive/MyDrive/sauti-sheng")
if not ARTIFACTS.is_dir():
    ARTIFACTS = Path("/content/audio-speech-ai/finetune-sauti-tts/artifacts")

print(ARTIFACTS)
display(Markdown((ARTIFACTS / "data/data_report.md").read_text(encoding="utf-8")))
"""
)

md(
    """
### Listen to the reference clip

The reference transcript is estimated from character share, not forced alignment. Play it and read
`ref_text.txt` before trusting the eval — a wrong ref text costs timbre fidelity, not intelligibility.
If it is off, re-cut with `!sauti-sheng data --ref-index 0 --force`.
"""
)

code(
    """
with_d = ARTIFACTS / "data"
print("ref text:", (with_d / "ref_text.txt").read_text(encoding="utf-8").strip())
print((with_d / "ref_meta.json").read_text(encoding="utf-8"))
display(Audio(str(with_d / "ref_audio.wav")))
"""
)

md(
    """
## 7. Stage 3 — extract the EMA init

The Sauti release is a **full training checkpoint** (weights + EMA + optimizer + scheduler +
`update: 15350`). Handing it to F5's `Trainer` as a `pretrained_*.pt` silently *resumes* it — optimizer
restored, LR already at its floor, nothing changes.

This stage keeps only `ema_model_state_dict` (~1.35 GB, Drive-safe, no `update` key) and deletes the
5.4 GB raw download from local scratch.
"""
)

code(
    """
!sauti-sheng model
"""
)

md(
    """
## 8. Stage 4 — train

Effective batch is 4 samples of 30 s on every profile, so epochs mean roughly the same amount of
training on each GPU:

| profile | GPU | batch × accum | epochs | updates |
|---|---|---|---|---|
| `l4` | L4 24 GB | 2 × 2 | 150 | 1650 |
| `a100` | A100 40/80 GB | 4 × 1 | 150 | 1650 |
| `t4` (needs `--allow-small-gpu`) | T4 16 GB | 1 × 4 | 200 | 2200 |

**Verify the wiring first** with a smoke run (1 epoch × 4 clips, ~2 min). It overwrites
`model_last.pt` with junk, so delete the checkpoint dir afterwards or just skip straight to the real
run on a fresh run name — see the note after the cell.
"""
)

code(
    """
# Optional wiring test. Writes a throwaway checkpoint: clear it before the real run.
# !sauti-sheng train --smoke
"""
)

code(
    """
# Real run. --max-hours stops cleanly at a checkpoint boundary before the runtime dies.
# Re-run this exact cell after a disconnect: it resumes from model_last.pt.
!sauti-sheng train --max-hours 8
"""
)

code(
    """
!sauti-sheng status
"""
)

md(
    """
If the smoke run left a checkpoint behind and you want a clean start:

```python
# !sauti-sheng train --fresh --max-hours 8   # deletes model_*.pt, keeps the EMA init
```

Checkpoint sizes: each training checkpoint is ~5 GB (weights + AdamW moments + EMA). `keep_last=3`
plus `model_last.pt` means ~22 GB on Drive — lower `--keep-last` if space is tight.
"""
)

md(
    """
## 9. Stage 5 — eval

Synthesises 3 hand-written Sheng probes plus 2 held-out dataset transcripts with the fine-tuned
checkpoint, and flags silence / runaway length / clipping. `--compare-base` also renders the
untouched Sauti model on the same prompts, so you hear what changed.

The default uses the **online** weights: early in a fine-tune the EMA lags behind.
"""
)

code(
    """
!sauti-sheng eval --compare-base
"""
)

code(
    """
from pathlib import Path
from IPython.display import Audio, display

evals = ARTIFACTS / "evals/sheng_abel"
runs = sorted((p for p in evals.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True) if evals.is_dir() else []

if not runs:
    print("no eval output yet - run the stage 5 cell first")
else:
    latest = runs[0]
    print("eval dir:", latest)
    print((latest / "eval_report.md").read_text(encoding="utf-8"))
    for wav in sorted(latest.rglob("*.wav")):
        print(wav.relative_to(latest))
        display(Audio(str(wav)))
"""
)

md(
    """
## 10. Stage 6 — export

Strips optimizer/EMA (~5 GB → 1.35 GB) and copies the weights, the vocab the checkpoint was trained
with, the reference clip and the eval samples into a self-contained folder that loads with stock
F5-TTS.
"""
)

code(
    """
!sauti-sheng export --zip
!sauti-sheng status
"""
)

code(
    """
from pathlib import Path
from google.colab import files

zip_path = ARTIFACTS / "exports/sheng_abel.zip"
if zip_path.is_file():
    print(zip_path, f"{zip_path.stat().st_size / 1024**3:.2f} GB")
    files.download(str(zip_path))
else:
    # .zip is gitignored and ~1.35 GB: download the folder contents instead
    files.download(str(ARTIFACTS / "exports/sheng_abel/model.pt"))
"""
)

md(
    """
## 11. One-shot (skip the stage-by-stage run)

From a fresh runtime, after install + Drive mount, this runs `setup → data → model → train → eval →
export`. Finished stages are skipped, training resumes, so it is safe to re-run after a disconnect.
"""
)

code(
    """
# !sauti-sheng all --max-hours 8
"""
)

md(
    """
## 12. Generate speech with the exported bundle

Self-contained: no repo needed, just `f5-tts` and the `exports/sheng_abel/` folder.

Keep prompts in the dataset's register — Swahili orthography with English loanwords. New vocabulary
will not be pronounced reliably, and the base voice is a single speaker.
"""
)

code(
    """
from f5_tts.api import F5TTS

exp = ARTIFACTS / "exports/sheng_abel"
tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file=str(exp / "model.pt"),
    vocab_file=str(exp / "vocab.txt"),   # must be this vocab, not Emilia's
    use_ema=False,
)

ref_text = (exp / "ref_text.txt").read_text(encoding="utf-8").strip()
wav, sr, _ = tts.infer(
    ref_file=str(exp / "ref_audio.wav"),
    ref_text=ref_text,
    gen_text="Niaje mazee, unafanya nini leo? Tunataka kupiga story kidogo na wewe.",
)
tts.export_wav(wav, "/content/niaje.wav")

from IPython.display import Audio, display
display(Audio("/content/niaje.wav"))
"""
)

md(
    """
## Troubleshooting

| symptom | fix |
|---|---|
| `no profile for … 0 GB VRAM` | No GPU. Change the runtime type. |
| `the t4 profile is tight (16 GB)` | Add `--allow-small-gpu` or switch to L4/A100. |
| `torchcodec` / `libav*` / ffmpeg error | Should not happen — `patch_torchaudio()` is applied by every stage. Check `ffmpeg -version`; set `SAUTI_KEEP_TORCHAUDIO=1` to see the real traceback. |
| Runtime died mid-training | Re-run the train cell. It resumes from `model_last.pt`. |
| Loss flat, output unchanged from base | Wrong init: `checkpoints/sheng_abel/pretrained_sauti.pt` must have no `update` key. Re-run `!sauti-sheng model`. |
| Babble / wrong characters | Vocab mismatch: `data/f5/vocab.txt` must be the 2545-entry Sauti vocab (`!sauti-sheng data --force`). |
| Drive full | `--keep-last 1`, and keep only `exports/` + `model_last.pt`. |
| pip broke CUDA | Restart the runtime, then re-run from the install cell. |
"""
)


notebook = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True, "gpuType": "L4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "colab_finetune_sauti_sheng.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8", newline="\n")
print("wrote", out, len(CELLS), "cells")
