# Fine-tune Sauti TTS to speak Sheng

Fine-tune [`msingiai/sauti-tts`](https://huggingface.co/msingiai/sauti-tts) — an F5-TTS v1 Base
checkpoint trained on Swahili — on **Sheng** (Nairobi Swahili + English code-switching).

Training data is [`lyle-mlengineer/kenyan-celebs`](https://huggingface.co/datasets/lyle-mlengineer/kenyan-celebs):
42 clips, 21 minutes, one speaker (`abel`). That is a **voice** fine-tune, not a language course.

Everything runs on a single Colab GPU. The CLI tracks stage progress on Google Drive, so a
runtime disconnect costs you the current step, not the run.

```bash
pip install -e .
pip install f5-tts==1.1.22      # pin: this recipe depends on Trainer/CustomDataset internals
sauti-sheng all                 # setup -> data -> model -> train -> eval -> export
```

---

## Contents

- [Notebook](#colab-notebook) — one click, stage by stage
- [Why this is not `f5-tts`'s own `finetune_cli.py`](#why-this-is-not-f5-ttss-own-finetune_clipy)
- [Requirements](#requirements)
- [Quick start (Colab)](#quick-start-colab)
- [Quick start (local / SSH)](#quick-start-local--ssh)
- [Stages](#stages)
- [CLI reference](#cli-reference)
- [Training profiles](#training-profiles)
- [Outputs](#outputs)
- [Running the fine-tuned model](#running-the-fine-tuned-model)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Layout](#layout)

---

## Colab notebook

[`notebooks/colab_finetune_sauti_sheng.ipynb`](notebooks/colab_finetune_sauti_sheng.ipynb) is the
full walkthrough: GPU check, Drive mount, install, then one cell per stage with listening checks
between them.

1. Open [Google Colab](https://colab.research.google.com/) → **File → Upload notebook** → pick the
   file above.
2. **Runtime → Change runtime type → L4 GPU** (A100 if you have Pro, T4 only with `--allow-small-gpu`).
3. **Runtime → Run all**, or run the cells one at a time and listen to `eval` output before
   exporting.

The notebook clones this repo into `/content`, installs it, and keeps every large artifact in
`/content/drive/MyDrive/sauti-sheng`. Re-running the notebook after a disconnect resumes: finished
stages are skipped and training continues from `model_last.pt`.

If you would rather drive it from a terminal / SSH session, the commands are identical to what the
notebook runs — see [Quick start (Colab)](#quick-start-colab).

---

## Why this is not `f5-tts`'s own `finetune_cli.py`

Three traps, each silent, each producing a run that looks fine and is not:

| Trap | What happens with plain `f5-tts` | What this repo does |
|---|---|---|
| The Sauti release is a **full training checkpoint** (`model_state_dict` + `ema_model_state_dict` + optimizer + scheduler + `update: 15350`). F5's `Trainer` chooses "resume" vs "initialise" by looking for `update`/`step` keys, **not** by filename. | You hand it as `pretrained_*.pt`, and it *resumes*: optimizer state restored, LR already decayed to its floor, step counter at 15350. Nothing changes, quietly. | `sauti-sheng model` extracts only `ema_model_state_dict` (~1.35 GB) into `pretrained_sauti.pt`, which has no `update` key, so it is an init. |
| The release ships its own 2545-entry `vocab.txt`. F5's prepare script copies **Emilia's** vocab. | Every token index means a different character than the checkpoint's text embedding expects. | `data` downloads Sauti's `vocab.txt` and uses that everywhere (dataset dir, training, inference). |
| On Colab, `torchaudio.load` delegates to `torchcodec`, which wants FFmpeg shared libs Colab does not have in the right place. | `libav*` / `torchcodec` crash in the middle of training or eval. | `compat.patch_torchaudio()` swaps `torchaudio.load/save/info` for `soundfile` (already a dependency). Set `SAUTI_KEEP_TORCHAUDIO=1` to disable. |

Two more decisions worth knowing:

- **Batching is `batch_size_type="sample"`.** Every clip is exactly 30 s = 2812 mel frames. F5's
  frame sampler drops samples over its frame budget, which would silently shrink the dataset.
- **Eval uses the online weights, not EMA.** Early in a fine-tune the EMA lags behind the online
  weights (`sauti-sheng eval --use-ema` forces EMA; `export --use-ema` does too).

---

## Requirements

- Python ≥ 3.10, CUDA GPU
- `f5-tts==1.1.22` (pulls torch, torchaudio, vocos, datasets, ema-pytorch, accelerate, transformers)
- Disk: the raw 5.4 GB release goes to **scratch** (local Colab disk, deleted after the EMA is
  extracted), so ~7 GB of local disk is enough. Training writes ~5 GB checkpoints to the **artifacts**
  root: budget ~25 GB of Drive for `keep_last=3` plus `model_last.pt`, or lower `--keep-last`.
- Network access to Hugging Face (dataset + checkpoint + Vocos) and Google Drive if on Colab.

Stage 1 (`sauti-sheng setup`) checks all of this and prints the install hint if something is missing.

---

## Quick start (Colab)

These are shell commands. In a notebook cell, prefix them with `!` and use `%cd` / `%pip` for the two
notebook builtins, exactly as [`notebooks/`](notebooks/colab_finetune_sauti_sheng.ipynb) does.

```bash
# 1. clone
cd /content
git clone --depth 1 https://github.com/lyle-mlengineer/audio-speech-ai.git
cd audio-speech-ai/finetune-sauti-tts

# 2. install
pip install -q -e .
pip install -q f5-tts==1.1.22

# 3. environment check (torch, F5, ffmpeg, GPU, VRAM, free disk)
sauti-sheng setup

# 4. artifacts must survive a disconnect: mount Drive in a notebook cell first
#    from google.colab import drive; drive.mount("/content/drive")
#    the CLI then auto-detects /content/drive/MyDrive/sauti-sheng as the artifacts root

# 5. run the pipeline
sauti-sheng all --max-hours 8
```

`--max-hours` stops at the next checkpoint boundary instead of being killed by the 12 h runtime
limit. Re-running the same command resumes.

Without Drive mounted, artifacts land in `<repo>/artifacts` and die with the VM — fine for a smoke
test, not for a real run.

---

## Quick start (local / SSH)

```bash
git clone https://github.com/lyle-mlengineer/audio-speech-ai.git
cd audio-speech-ai/finetune-sauti-tts
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pip install f5-tts==1.1.22

# optional: choose where artifacts go
export SAUTI_SHENG_ARTIFACTS=/data/sauti-sheng
export SAUTI_SHENG_SCRATCH=/tmp/sauti-scratch   # fast local disk for the 5.4 GB raw ckpt

sauti-sheng setup
sauti-sheng all --max-hours 6
```

`sauti-sheng` is installed as a console script by `pip install -e .`; `python -m sauti_sheng.cli`
works without it.

---

## Stages

Each stage is idempotent and independently runnable. State lives in
`<artifacts>/progress/state.json` and is mirrored into `progress.md`.

| # | Command | Does | Output |
|---|---|---|---|
| 1 | `setup` | Reports Python/torch/F5/vocos/ffmpeg/GPU/VRAM/free disk, picks a training profile | `progress/state.json` |
| 2 | `data` | Downloads the parquet + Sauti vocab, resamples 22.05 kHz → 24 kHz, peak-normalises, writes 42 WAVs, the F5 `raw.arrow`/`duration.json`, train/val splits, and a ~9 s reference clip cut at a pause | `data/`, `data/data_report.md`, `data/ref_audio.wav`, `data/ref_text.txt` |
| 3 | `model` | Extracts the EMA weights from the 5.4 GB release, checks the text embedding against the vocab, deletes the raw file | `pretrained/sauti-tts/sauti_ema.pt`, `model_report.md` |
| 4 | `train` | Wraps F5's `Trainer`: explicit dataset, vocab, init and resume paths | `checkpoints/<run>/model_last.pt`, `model_<update>.pt` |
| 5 | `eval` | Synthesises 3 Sheng probes + 2 held-out val transcripts; flags silence, runaway length, clipping. `--compare-base` also renders the untouched Sauti model | `evals/<run>/<ckpt>/`, `eval_report.md` |
| 6 | `export` | Strips optimizer/EMA (~5 GB → 1.35 GB), copies vocab + reference clip + samples, writes a model card, optional `--zip` | `exports/<run>/` |

`all` runs 1→6 and skips stages already marked done. `--no-resume` redoes them (training still
resumes from `model_last.pt` unless you pass `--fresh`).

Check progress any time:

```bash
sauti-sheng status
```

```
run: sheng_abel   (updated 2025-01-02 11:04:33)
 [x] 1. setup    Environment (torch + F5-TTS) ready  --  NVIDIA L4 / l4
 [x] 2. data     ... -- 42 clips / 21.0 min @ 24000 Hz -> 38 train + 4 val
 [x] 3. model    ... -- EMA 1.35 GB (raw 5.39 GB deleted)
 [~] 4. train    F5-TTS fine-tune on Sheng  --  150 epochs, batch 2x2 @ l4
 [ ] 5. eval     Checkpoint synthesised + sanity checked
 [ ] 6. export   Self-contained model dir + samples packaged
```

---

## CLI reference

```
sauti-sheng [--artifacts PATH] [--scratch PATH] [--run NAME] {status,setup,data,model,train,eval,export,all}
```

Global flags:

| flag | meaning |
|---|---|
| `--artifacts PATH` | artifacts root. Default order: flag → `$SAUTI_SHENG_ARTIFACTS` → `/content/drive/MyDrive/sauti-sheng` (if Drive is mounted) → `<repo>/artifacts` |
| `--scratch PATH` | fast local disk for the raw 5.4 GB checkpoint and parquet. Default: `$SAUTI_SHENG_SCRATCH` → `/content/sauti-sheng-scratch` → `<artifacts>/.scratch` |
| `--run NAME` | run name, default `sheng_abel`. One run dir per experiment. |

**data**

| flag | default | meaning |
|---|---|---|
| `--val-size` | `4` | held-out clips |
| `--seed` | `42` | split seed |
| `--ref-index` | auto | force a specific val clip as the reference |
| `--ref-seconds` | `9.0` | target reference length; cut is moved to the nearest pause |
| `--force` | off | redo from the parquet |

**train**

| flag | default | meaning |
|---|---|---|
| `--profile {t4,l4,a100}` | auto-detect | see [profiles](#training-profiles) |
| `--allow-small-gpu` | off | required to accept the T4 profile |
| `--epochs`, `--batch-size`, `--grad-accum`, `--lr` | profile | override any profile value |
| `--save-interval` / `--last-interval` | profile | numbered checkpoint / `model_last.pt` cadence, in updates |
| `--keep-last` | profile | numbered checkpoints kept |
| `--mixed-precision {bf16,fp16,no}` | profile | |
| `--num-workers` | `2` | dataloader workers |
| `--max-hours` | `0` (off) | stop cleanly at the next checkpoint boundary |
| `--limit` | — | cap the number of training clips (debug) |
| `--fresh` | off | delete `model_*.pt` and restart from the Sauti EMA init |
| `--smoke` | off | 1 epoch × 4 clips × 1 sample/step — proves wiring, then throw it away |

**eval**

| flag | default | meaning |
|---|---|---|
| `--checkpoint` | `<run>/model_last.pt` | checkpoint to evaluate |
| `--use-ema` | off | evaluate EMA weights (default: online weights) |
| `--compare-base` | off | also render the untouched Sauti model |
| `--nfe-step` | `32` | diffusion steps |
| `--device` | `cuda` | |

**export**: `--checkpoint`, `--use-ema`, `--zip`.

---

## Training profiles

Effective batch is **4 samples everywhere**, so `epochs` means roughly the same amount of training
on each GPU. One sample is a 30 s clip = 2812 mel frames; F5's VRAM formula allows ~9728 frames on
24 GB, so two samples per forward pass fit comfortably.

| profile | GPU | batch × accum | epochs | updates | precision |
|---|---|---|---|---|---|
| `l4` (default) | L4 24 GB | 2 × 2 | 150 | 1650 | bf16 |
| `a100` | A100 40/80 GB | 4 × 1 | 150 | 1650 | bf16 |
| `t4` | T4 16 GB | 1 × 4 | 200 | 2200 | fp16 |

The profile is auto-detected from the GPU name and VRAM; `--profile` overrides. T4 refuses to run
without `--allow-small-gpu` because it is genuinely tight. `configs/*.yaml` are human-readable
templates only — nothing reads them back. The settings actually used are rendered to
`<artifacts>/configs/<run>.yaml` on every `train`, so a run is reproducible after a disconnect.

---

## Outputs

```
<artifacts>/                     # Google Drive on Colab
├── data/
│   ├── wav24k/utt_0001.wav …    # 42 × 24 kHz mono PCM16
│   ├── f5/{raw.arrow,duration.json,vocab.txt}
│   ├── metadata_{train,val}.csv
│   ├── ref_audio.wav, ref_text.txt, ref_meta.json
│   └── data_report.md           # charset vs vocab, pinyin identity, ref-clip confidence
├── pretrained/sauti-tts/
│   ├── sauti_ema.pt             # 1.35 GB init checkpoint
│   └── model_report.md
├── configs/sheng_abel.yaml      # rendered settings for the run
├── checkpoints/sheng_abel/
│   ├── pretrained_sauti.pt      # the init, hardlinked/copied
│   ├── model_last.pt            # ~5 GB: weights + optimizer + EMA
│   └── model_<update>.pt
├── evals/sheng_abel/model_last/ # 5 prompts, finetuned/ + base/ (with --compare-base)
├── exports/sheng_abel/          # model.pt, vocab.txt, ref_audio.wav, ref_text.txt,
│                                # config.json, README.md, samples/
└── progress/{state.json,progress.md}
```

Note that training checkpoints are ~5 GB each (weights + AdamW moments + EMA). With
`keep_last=3` plus `model_last.pt`, budget ~22 GB of Drive. `export` is the 1.35 GB artifact you
actually want to keep.

---

## Running the fine-tuned model

`exports/<run>/` is self-contained and loads with stock F5-TTS — no repo needed:

```python
from f5_tts.api import F5TTS

tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file="exports/sheng_abel/model.pt",
    vocab_file="exports/sheng_abel/vocab.txt",   # must be this vocab, not Emilia's
    use_ema=False,
)
wav, sr, _ = tts.infer(
    ref_file="exports/sheng_abel/ref_audio.wav",
    ref_text=open("exports/sheng_abel/ref_text.txt", encoding="utf-8").read().strip(),
    gen_text="Niaje mazee, unafanya nini leo? Tunataka kupiga story kidogo na wewe.",
)
tts.export_wav(wav, "out.wav")
```

Within this repo, `sauti_sheng.infer` does the same thing without pulling `transformers`, `pydub`,
`matplotlib` or the ASR pipeline (`infer.py` reproduces the handful of upstream pieces that matter:
Vocos loading, sentence chunking, the duration heuristic, cross-fading):

```python
from sauti_sheng.infer import load_reference, load_vocoder, synthesise, write_wav
from sauti_sheng.model import load_for_inference
from sauti_sheng.paths import Paths, resolve_artifacts

paths = Paths(resolve_artifacts())
model, _ = load_for_inference(paths, paths.latest_checkpoint("sheng_abel"), use_ema=False)
vocoder = load_vocoder("cuda")
ref, rate, ref_text = load_reference(paths)
wave = synthesise(model, vocoder, ref, ref_text, "Weka pesa kwa M-Pesa, alafu nitakutumia confirmation message.")
write_wav(paths.artifacts / "out.wav", wave)
```

Practical limits:

- 21 minutes is a voice fine-tune. Keep prompts in the dataset's register — Swahili orthography with
  English loanwords. New vocabulary will not be pronounced reliably.
- Single speaker. The reference clip steers timbre only; it is not a speaker-conditioned model.
- The reference transcript is estimated from character share, not forced alignment. A wrong ref text
  costs timbre fidelity, not intelligibility.
- Check the rights of the source material before publishing a cloned voice.

---

## Tests

The CLI, data staging, config and checkpoint logic are covered without a GPU or `f5-tts`:

```bash
pip install -e '.[dev]'
pytest -q
```

`tests/` covers vocab/charset handling, resampling, the 22.05→24 kHz path, split determinism,
profile selection and VRAM floors, progress state, checkpoint init-vs-resume detection, and the
torchaudio compat shim.

---

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `no profile for … with 0 GB VRAM` | No GPU. **Runtime → Change runtime type → L4 GPU**. |
| `the t4 profile is tight (16 GB)` | Add `--allow-small-gpu`, or switch to L4/A100. |
| `torchcodec` / `libavcodec` / `ffmpeg` error | Should not happen: `patch_torchaudio()` is applied by every stage. If it does, check `ffmpeg -version`, or set `SAUTI_KEEP_TORCHAUDIO=1` to see the real traceback. |
| Colab runtime died mid-training | Just re-run: `sauti-sheng all` resumes from `model_last.pt` (`--last-interval` controls how much is lost). |
| Training loss flat, output unchanged after 150 epochs | Classic symptom of resuming Sauti's checkpoint instead of initialising from it. Check that `checkpoints/<run>/pretrained_sauti.pt` has no `update` key and `model_report.md` says EMA-only. |
| Output is babble / wrong characters | Vocab mismatch. The text embedding must match `data/f5/vocab.txt` (2545 entries, copied from the Sauti repo). |
| `missing pretrained checkpoint …` | Run `sauti-sheng model`. |
| `FileNotFoundError: …/raw.arrow` | Run `sauti-sheng data`. |
| Drive fills up | Lower `--keep-last`, and keep only `exports/` plus `model_last.pt`. |
| `pip install f5-tts` changed torch and CUDA broke | Restart the runtime after installing, then re-run `sauti-sheng setup`. |
| Evaluation says "suspicious" | Listen before exporting: silence, runaway length or clipping. Try `--use-ema` or an earlier `model_<update>.pt`. |

---

## Layout

```
finetune-sauti-tts/
├── src/sauti_sheng/
│   ├── cli.py         # sauti-sheng <stage>, progress bookkeeping
│   ├── paths.py       # where things live; artifacts/scratch resolution
│   ├── config.py      # GPU profiles, rendered YAML
│   ├── env.py         # stage 1: can this machine do the job?
│   ├── data.py        # stage 2: parquet -> WAVs, F5 dataset, reference clip
│   ├── resample.py    # 22.05 -> 24 kHz without scipy
│   ├── text.py        # cleaning, charset vs vocab, probe prompts
│   ├── model.py       # stage 3: checkpoint -> EMA init; model builder/loader
│   ├── checkpoint.py  # init-vs-resume detection, EMA/online selection, export
│   ├── train.py       # stage 4: thin wrapper around f5_tts.model.Trainer
│   ├── eval.py        # stage 5: probes + held-out prompts + sanity metrics
│   ├── export.py      # stage 6: weights-only bundle + model card
│   ├── infer.py       # F5 sampling without upstream's heavy imports
│   ├── compat.py      # torchaudio -> soundfile shim, seeding
│   └── progress.py    # stage state on Drive + progress.md mirror
├── configs/           # human-readable profile templates (never loaded)
├── notebooks/         # Colab walkthrough
├── tests/
├── pyproject.toml
└── requirements.txt
```

## References

- [F5-TTS](https://github.com/SWivid/F5-TTS) — `F5TTS_v1_Base`, `Trainer`, `CustomDataset`
- [`msingiai/sauti-tts`](https://huggingface.co/msingiai/sauti-tts) — base checkpoint + vocab
- [`lyle-mlengineer/kenyan-celebs`](https://huggingface.co/datasets/lyle-mlengineer/kenyan-celebs) — 42 clips, speaker `abel`
- [`charactr/vocos-mel-24khz`](https://huggingface.co/charactr/vocos-mel-24khz) — 24 kHz vocoder

## License

See the repository root [`LICENSE`](../LICENSE). The base model and dataset belong to their
respective authors; check their terms before redistributing weights or synthesised audio.
