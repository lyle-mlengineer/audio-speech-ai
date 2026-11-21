"""Stage 7: package the fine-tune so it can be loaded without this repo.

F5 training checkpoints contain the optimizer, scheduler and EMA - ~5 GB for a
model that needs 1.35 GB. The export keeps only the weights, alongside the vocab
the checkpoint was trained with, the reference clip used for voice cloning, and
the eval samples.

The result loads with stock F5-TTS:

.. code-block:: python

    from f5_tts.api import F5TTS
    tts = F5TTS(model="F5TTS_v1_Base", ckpt_file="model.pt",
                vocab_file="vocab.txt", use_ema=False)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .checkpoint import export_state, select_state_dict
from .paths import Paths, checkpoint_update

CARD = """\
# Sauti TTS - Sheng fine-tune ({run})

F5-TTS v1 Base, initialised from [`msingiai/sauti-tts`](https://huggingface.co/msingiai/sauti-tts)
and fine-tuned on `lyle-mlengineer/kenyan-celebs` (42 clips, 21 min, single
speaker `abel`, Sheng / Nairobi Swahili + English).

## Files

| file | what |
|---|---|
| `model.pt` | weights only (no optimizer/EMA) |
| `vocab.txt` | 2545-entry character vocab - **must** be used with this model |
| `ref_audio.wav` | ~9 s reference clip for voice cloning |
| `ref_text.txt` | transcript of the reference clip (estimated from character share) |
| `config.json` | F5 model + mel settings |
| `samples/` | eval renders, if `sauti-sheng eval` was run |

## Use

```python
from f5_tts.api import F5TTS

tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file="model.pt",
    vocab_file="vocab.txt",
    use_ema={use_ema},
)
wav, sr, _ = tts.infer(
    ref_file="ref_audio.wav",
    ref_text=open("ref_text.txt", encoding="utf-8").read().strip(),
    gen_text="Niaje mazee, unafanya nini leo?",
)
tts.export_wav(wav, "out.wav")
```

## Limits

* 21 minutes is a **voice** fine-tune, not a language course. Keep prompts in
  the dataset's register (Swahili orthography + English loanwords).
* single speaker.
* the reference transcript is estimated from character share, not forced
  alignment - a wrong ref text costs timbre fidelity, not intelligibility.
* check the rights of the source material before publishing the voice.
"""


def _update_number(path: Path) -> int | None:
    return checkpoint_update(path)


def export(
    paths: Paths,
    run: str,
    *,
    checkpoint: Path | str | None = None,
    use_ema: bool = False,
    zip_bundle: bool = False,
) -> Path:
    import torch

    checkpoint = Path(checkpoint or paths.latest_checkpoint(run))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} missing - train first, or pass --checkpoint")

    dest = paths.export_dir(run)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = export_state(raw, use_ema=use_ema)
    torch.save(state, dest / "model.pt")
    online = select_state_dict(raw, use_ema=use_ema)
    n_params = sum(int(v.numel()) for v in online.values() if hasattr(v, "numel"))
    del raw, state, online

    for src, name in (
        (paths.sauti_vocab, "vocab.txt"),
        (paths.ref_audio, "ref_audio.wav"),
        (paths.ref_text_file, "ref_text.txt"),
    ):
        if src.is_file():
            shutil.copy2(src, dest / name)

    vocab_path = dest / "vocab.txt"
    vocab_size = (
        sum(1 for line in vocab_path.read_text(encoding="utf-8").splitlines() if line)
        if vocab_path.is_file()
        else None
    )

    config = {
        "model": "F5TTS_v1_Base",
        "init": "msingiai/sauti-tts",
        "run": run,
        "source_checkpoint": checkpoint.name,
        "update": _update_number(checkpoint),
        "use_ema": use_ema,
        "parameters": n_params,
        "sample_rate": 24000,
        "mel": {"n_fft": 1024, "hop_length": 256, "win_length": 1024, "n_mel_channels": 100},
        "vocab_size": vocab_size,
    }
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (dest / "README.md").write_text(CARD.format(run=run, use_ema=use_ema), encoding="utf-8")

    samples = paths.eval_dir(run) / checkpoint.stem
    if samples.is_dir():
        target = dest / "samples"
        target.mkdir()
        for wav in sorted(samples.glob("*.wav")):
            shutil.copy2(wav, target / wav.name)
        if (samples / "eval_report.md").is_file():
            shutil.copy2(samples / "eval_report.md", target / "eval_report.md")

    if zip_bundle:
        archive = shutil.make_archive(str(dest), "zip", root_dir=dest.parent, base_dir=dest.name)
        return Path(archive)
    return dest
