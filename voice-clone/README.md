# sheng-voiceclone

Fine-tune a **voice cloning** TTS model for **Sheng** (Nairobi Swahili + English)
on a free **Colab GPU**, using `lyle-mlengineer/kenyan-celebs` (42 clips x 30 s,
one speaker, ~21 min).

The model is **Chatterbox Multilingual** (`ResembleAI/chatterbox`, MIT, 0.5 B
`T3` LM). `sw` (Swahili) is already in its `SUPPORTED_LANGUAGES`, and it clones any
voice zero-shot from a ~6 s reference clip - so the fine-tune only has to learn
*this speaker's Sheng delivery*, not Swahili from scratch.

This directory is the **measure-first** layer: it prepares data, clones the
baseline, fine-tunes, and then **scores both runs** so you can tell whether
training helped or just made the audio worse. Training itself is delegated to the
sibling project (`../finetune-chatterbox`), so there is one trainer, not two.

```
setup -> data -> model -> baseline -> train -> eval -> compare
```

---

## Why this model (and not XTTS / F5 / CosyVoice)

| candidate | Swahili? | zero-shot cloning? | fine-tunable T3/LM? |
|---|---|---|---|
| **Chatterbox Multilingual** | **yes (`sw`, 23 langs)** | yes, 6 s reference | yes (T3, LoRA or full) |
| XTTS-v2 (Coqui) | no (17 langs) | yes | yes |
| F5-TTS / CosyVoice2 / IndexTTS2 | no (zh/en + a few) | yes | yes |
| VoxCPM2 | unclear | yes | yes (LoRA, sibling project) |
| Qwen3-TTS | claims sw | yes | yes (sibling project) |

Sheng is a Swahili matrix with English content words, so the base model must at
minimum speak Swahili. That leaves Chatterbox. There is precedent: Msingi-AI's
[`msingiai/sauti_tts_v2`](https://huggingface.co/msingiai/sauti_tts_v2) is a
Swahili Chatterbox **LoRA** with published CER/UTMOS numbers, and the community
[`chatterbox-finetuning`](https://github.com/gokhaneraslan/chatterbox-finetuning)
kit is what it was trained with.

---

## The whole thing in 5 cells

```python
# 1. L4 runtime, then
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ['SHENG_VOICECLONE_ARTIFACTS'] = '/content/drive/MyDrive/voice-clone-sheng'

# 2. code + deps
!git clone https://github.com/<you>/audio-speech-ai.git /content/audio-speech-ai
%cd /content/audio-speech-ai/voice-clone
!pip install -q -e . && pip install -q -U chatterbox-tts transformers speechbrain
!pip install -q -e ../finetune-chatterbox

# 3. environment + GPU profile
!sheng-voiceclone setup

# 4. dataset -> 16 kHz wavs, splits, reference clip
!sheng-voiceclone data && sheng-voiceclone model

# 5. the number to beat, then the tune, then the verdict
!sheng-voiceclone baseline --prompts sheng,swahili
!sheng-voiceclone train --profile l4 --steps 600 --lr 5e-5 --lora --max-hours 8
!sheng-voiceclone eval --prompts sheng,swahili && sheng-voiceclone compare
```

`notebooks/voice_clone_sheng_colab.ipynb` is exactly that, cell by cell, with
audio playback and the after-disconnect cell.

---

## The decision that matters: measure before you train

A cloned clip fails in one of three ways, and only one of them is fixed by
training:

| failure | how it is caught | fixed by |
|---|---|---|
| **silence / clipping / never ends** | RMS, clipped fraction, duration vs text length - **no model needed** | nothing to train: bad inference or LR |
| **wrong words** (hallucination, truncation) | CER/WER of the ASR transcript vs the prompt | less training, lower LR |
| **wrong voice** (right words, not the speaker) | cosine of ECAPA speaker embeddings vs the reference clip | more/better training |

`baseline` scores the **untuned** model on 7 fixed prompts (4 Sheng code-switched,
2 clean Swahili, 1 English). `eval` scores the checkpoint on the *same* prompts.
`compare` prints `KEEP` / `MIXED` / `REJECT` / `NO GAIN`, e.g.:

```
| metric      | baseline | tuned | delta  | better   |
| ok clips    | 6/7      | 6/7   | +0     | higher   |
| CER mean    | 0.412    | 0.377 | -0.035 | lower    |
| speaker sim | 0.581    | 0.734 | +0.153 | higher   |
Verdict: KEEP: closer to the speaker without losing the words.
```

Numbers are only meaningful **relative** to your own baseline. Sheng is
code-switched and Swahili is agglutinative, so absolute CER is high for every
model, including the base one.

---

## Stage reference

| stage | command | does | needs GPU |
|---|---|---|---|
| 1 | `sheng-voiceclone setup` | torch / cuda / deps check, GPU -> profile, renders `configs/<run>.yaml` | - |
| 2 | `sheng-voiceclone data` | 42 clips -> 16 kHz mono PCM16 WAVs, 34 train / 4 val / 4 test, `ref_audio.wav` from **val**, `data_report.md` | - |
| 3 | `sheng-voiceclone model` | `ResembleAI/chatterbox` snapshot (`t3`/`ve`/`s3gen`/`conds`/tokenizer) | - |
| 4 | `sheng-voiceclone baseline` | zero-shot clone + CER/WER/speaker-sim scores | yes |
| 5 | `sheng-voiceclone train` | T3 fine-tune (LoRA or full) via `../finetune-chatterbox` | yes |
| 6 | `sheng-voiceclone eval` | same prompts through the checkpoint + scores | yes |
| 7 | `sheng-voiceclone compare` | baseline vs tuned table + verdict | - |
| - | `sheng-voiceclone score --dir NAME` | re-score an eval dir (e.g. with a bigger ASR judge) | yes |
| - | `sheng-voiceclone status` | what is done, what is next, current scores | - |
| - | `sheng-voiceclone all --resume` | every stage, skipping the done ones | yes |

Everything downstream of `data` is idempotent: existing WAVs are reused, finished
stages are skipped with `--resume`.

### Profiles

| profile | GPU | batch x accum | dtype |
|---|---|---|---|
| `l4` | Colab L4 24 GB (default) | 4 x 2 | bfloat16 |
| `a100` | Colab A100 | 8 x 1 | bfloat16 |
| `t4` | Colab T4 16 GB | 2 x 4 | float16 |

Effective batch is **8** everywhere, so `--steps` and `--lr` mean the same thing
on every GPU:

```bash
!sheng-voiceclone train --profile l4 --steps 600 --lr 5e-5 --lora --max-hours 8
```

Default is a full T3 fine-tune at `5e-5`; `--lora` switches to adapters, which is
the right call for 21 minutes of audio (the community kit's own guidance: LoRA for
anything under ~10 h). Either way the LR is low on purpose - the reported failure
mode for this family at `2e-5`+ is noise or endless generation.

---

## Use your own voice instead of the dataset

```bash
# /my/recordings/  *.wav + *.txt sidecars (or transcripts.csv: file,text)
!sheng-voiceclone data --source /content/my-voice --val-clips 2 --test-clips 2
```

Then run the same stages. The reference clip is chosen automatically from the
validation split (longest clip, never trained on).

---

## Where things live

`$SHENG_VOICECLONE_ARTIFACTS`, else `/content/drive/MyDrive/voice-clone-sheng` on
Colab, else `./artifacts`.

```
<artifacts>/
  data/wav16k/utt_0001.wav ...   data/ref_audio.wav   data/data_report.md
  manifests/train.jsonl  val.jsonl  test.jsonl
  pretrained/chatterbox/            # t3_mtl23ls_v2.safetensors + ve/s3gen/conds
  checkpoints/<run>/latest/         # what `eval --resume` picks up
  evals/baseline/*.wav  evals/tuned_<step>/*.wav
  scores/baseline.{json,md}  tuned_<step>.{json,md}  compare.md
  configs/<run>.yaml   progress/state.json   progress/runs.jsonl
```

Budget: base model ~2 GB, ~1 GB per checkpoint (keep 3), eval WAVs ~50 MB.

## After a disconnect

```python
%cd /content/audio-speech-ai/voice-clone
!git pull
!pip install -q -e . && pip install -q -e ../finetune-chatterbox
!sheng-voiceclone status                  # what is done, what is left
!sheng-voiceclone all --resume --max-hours 8
```

`progress.md` (repo root and artifacts root) always shows the stage list and the
one command to run next.

---

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `ERROR: no reference clip ... run \`data\` first` | stage order; run `sheng-voiceclone data` |
| `trainer not installed` from `train` | `pip install -e ../finetune-chatterbox` on the same VM |
| `ERROR: no checkpoint under .../checkpoints` | training has not saved yet, or `--run` differs from the trained run |
| all clips `silent` | wrong language id, or the checkpoint does not fit the T3 (`eval` says how many keys were missing) |
| all clips `runaway` | LR too high / too many steps - `--lr 2e-5`, fewer steps |
| `status: wrong-voice` after tuning | overfit to the training clips; use the earlier checkpoint |
| OOM on T4 | `--profile t4` (and keep `batch_size * grad_accum = 8`) |
| CER looks terrible everywhere | expected on code-switched Sheng; compare deltas, or re-score with `--asr-model openai/whisper-large-v3` |

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 110 tests, no GPU, no network
```

They cover the polyphase resampler (unity gain), manifest validation, split and
reference-clip invariants, the scorer's verdict ordering, profile/YAML rendering,
progress resume, and the offline CLI paths (`data --source`, `baseline --dry-run`,
`score`, `compare`).

## Honest limits

* 21 minutes is a **voice** fine-tune. New Sheng slang the base model has never
  seen will not suddenly be pronounced correctly - keep prompts in the dataset's
  register (Swahili orthography + English content words).
* Single speaker. Multi-speaker SFT is not supported by this recipe.
* Chatterbox is **archived research** (MIT, unmaintained since Aug 2026); the
  weights and code stay public, but nothing gets fixed upstream.
* The dataset is derived from third-party video - check the rights before
  publishing an exported voice.
