# sheng-qwentts

Fine-tune **`Qwen/Qwen3-TTS-12Hz-0.6B-Base`** to speak **Sheng** (Nairobi Swahili +
English slang) on a **Colab GPU**, using **`lyle-mlengineer/kenyan-celebs`**
(42 clips, 21 min, single speaker `abel`).

It is the official single-speaker SFT recipe from
[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) with the
things that actually break on 0.6B and on Colab fixed, wrapped in a driver that
resumes after a disconnect.

```
data -> model -> codes -> train -> eval -> export
```

---

## TL;DR - the whole thing in 4 cells

```python
# 1. GPU + Drive
!nvidia-smi
from google.colab import drive; drive.mount('/content/drive')

# 2. code
!git clone https://github.com/<you>/audio-speech-ai.git /content/audio-speech-ai
%cd /content/audio-speech-ai/finetune-qwentts

# 3. deps (qwen-tts pins transformers==4.57.3 - let it)
!pip install -q -e . && pip install -q -U qwen-tts
!sheng-qwentts setup

# 4. everything, resumable, stop-safe
!sheng-qwentts all --resume --profile l4 --max-hours 8
```

`notebooks/qwentts_sheng_colab.ipynb` is exactly that, cell by cell, with audio
playback and a "say something yourself" cell.

GPU: **L4 (24 GB)** is the sweet spot; **A100** is faster (`--profile a100`,
needs `flash-attn`); **T4 (16 GB)** works with `--profile t4 --allow-small-gpu`.

---

## What is actually being trained

Qwen3-TTS-Base is a discrete multi-codebook LM: text tokens + 16 codebooks at
12 Hz, decoded to 24 kHz audio by `Qwen3-TTS-Tokenizer-12Hz`.

Single-speaker fine-tuning trains **only the talker** (the LM), and bakes the
speaker into the checkpoint:

* the reference clip goes through the ECAPA **speaker encoder** -> one vector;
* that vector is injected at the `<speaker>` slot of every training sequence;
* at save time it is written into `talker.model.codec_embedding.weight[3000]`,
  and `config.json` gets `tts_model_type: custom_voice` +
  `talker_config.spk_id: {abel: 3000}`.

So the result is a **CustomVoice** model: no reference audio at inference, just
`generate_custom_voice(text, speaker="abel")`.

What 21 minutes of audio buys you: the **voice, accent and Sheng delivery** of
that speaker. Not new vocabulary - Qwen3-TTS already speaks Swahili and English.

---

## The things that break (all handled here)

| # | Problem | What this repo does |
|---|---|---|
| 1 | **Upstream `sft_12hz.py` crashes on 0.6B**: `text_hidden_size` 2048 != `hidden_size` 1024, so `text_embedding + codec_embedding` fails ([#198](https://github.com/QwenLM/Qwen3-TTS/issues/198), [#335](https://github.com/QwenLM/Qwen3-TTS/issues/335)). Inference already routes text through `talker.text_projection`; training did not. A projection with bias also leaks into padded positions unless the mask is re-applied. | `src/sheng_qwentts/embeds.py`, unit-tested against fake 2048/1024 tensors (`tests/test_embeds.py`) |
| 2 | **`--init_model_path` must be a directory**: the upstream trainer does `shutil.copytree(MODEL_PATH, out)` per checkpoint, so a `Qwen/...` repo id fails. | stage `model` snapshots both repos into the artifacts root first |
| 3 | **24 kHz only**: `finetuning/dataset.py` asserts `sr == 24000`; the dataset is 22.05 kHz (its card claims 16 kHz, it is wrong). | stage `data` resamples once, numpy polyphase (scipy when present), verified by tests |
| 4 | **Bad default LR**: 2e-5 / 1e-5 / 2e-6 are all reported to give noise or endless generation ([#39](https://github.com/QwenLM/Qwen3-TTS/issues/39)). | default `2e-6` with warmup + cosine decay, gradient clipping, and an eval stage that flags silent/noisy output instead of letting you find out by ear |
| 5 | **No resume, no mid-epoch checkpoints**: Colab dies, the run is gone. | step-based checkpoints + `latest.txt` pointer (no 1.4 GB copy per save), `--resume`, `--max-hours` |
| 6 | **`flash_attention_2` needs a 10-20 min source build** on Colab. | `sdpa` (PyTorch native) is the default; flash-attn is opt-in via `--profile a100` and falls back to sdpa if missing |
| 7 | **Every line needs the same `ref_audio`** or the speaker embedding drifts. | stage `data` picks one training clip and points all lines at `data/ref_audio.wav`; the trainer refuses mixed `ref_audio` |

---

## Pipeline

| stage | command | does | needs GPU |
|---|---|---|---|
| 1 | `sheng-qwentts setup` | torch / CUDA / qwen-tts / transformers check, GPU -> profile | - |
| 2 | `sheng-qwentts data` | parquet -> 42 x 24 kHz mono WAVs, 38 train / 4 val manifests, one shared `ref_audio.wav`, `data_report.md` | - |
| 3 | `sheng-qwentts model` | snapshots `Qwen3-TTS-12Hz-0.6B-Base` (~0.9 GB) + `Qwen3-TTS-Tokenizer-12Hz` (~0.17 GB) | - |
| 4 | `sheng-qwentts codes` | `audio_codes` (12 Hz x 16 codebooks) -> `train_with_codes.jsonl` | yes |
| 5 | `sheng-qwentts train` | SFT of the talker, 240 steps, resumes from `latest.txt` | yes |
| 6 | `sheng-qwentts eval` | synthesises 3 Sheng probes + 2 held-out transcripts, flags silence/noise, optional Base-model A/B | yes |
| 7 | `sheng-qwentts export` | self-contained model dir + samples + model card | - |

`all` chains them and records progress in `progress.md` and
`<artifacts>/progress/state.json`. Every stage is idempotent.

## Profiles

| profile | GPU | batch x accum | dtype | attention |
|---|---|---|---|---|
| `l4` | Colab L4 24 GB | 2 x 2 | bfloat16 | sdpa |
| `a100` | Colab A100 | 4 x 1 | bfloat16 | flash_attention_2 |
| `t4` | Colab T4 16 GB (`--allow-small-gpu`) | 1 x 4 | float16 | sdpa |

Effective batch is **4** everywhere, so `--steps` means the same thing on every
GPU. Finer control:

```bash
!sheng-qwentts train --profile l4 --steps 320 --lr 1e-6 --save-interval 40 --max-hours 6
```

Rendered settings land in `<artifacts>/configs/<run>.yaml`; checkpoints in
`<artifacts>/checkpoints/<run>/step_000XXXX/`.

## Where things live

On Colab the artifacts root is `/content/drive/MyDrive/qwentts-sheng` (Drive,
survives the VM). Locally it is `./artifacts`. Override with `--artifacts PATH`
or `$SHENG_QWENTTS_ARTIFACTS`.

```
<artifacts>/
  data/wav24k/utt_0001.wav ...   data/ref_audio.wav   data/data_report.md
  manifests/train_raw.jsonl  train_with_codes.jsonl  val_raw.jsonl  val_with_codes.jsonl
  pretrained/qwen3-tts-0.6b-base/  pretrained/qwen3-tts-tokenizer-12hz/
  checkpoints/sheng_06b/step_0000040/ (config.json + model.safetensors + speech_tokenizer/ + train_state.json)
                         latest.txt
  evals/sheng_06b/step_0000240/ft_01.wav ...
  exports/sheng_06b/model/ README.md samples/
```

Budget: Base 1.1 GB + tokenizer 0.17 GB + ~1.4 GB per checkpoint (keep 3) +
eval wavs. ~6 GB of Drive for a full run.

## After a disconnect

```python
%cd /content/audio-speech-ai/finetune-qwentts
!git pull
!pip install -q -e . && pip install -q -U qwen-tts
!sheng-qwentts status                    # what is done, what is left
!sheng-qwentts all --resume --max-hours 8
```

Training continues from `latest.txt`. Optimizer moments are not saved (they
would be 4.8 GB per checkpoint); AdamW re-warms within a few dozen steps, so
resume is safe but not bit-identical.

## Evaluating honestly

`!sheng-qwentts eval --compare-base` gives you, in `evals/<run>/<ckpt>/`:

* `ft_01..05.wav` - the fine-tuned voice (3 hand-written Sheng probes + 2
  held-out dataset transcripts);
* `base_clone_01..05.wav` - the untouched Base model cloning from `ref_audio`
  (so you hear what the fine-tune changed, not just what it sounds like);
* `eval_report.md` with duration / RTF / RMS and a verdict per file.

Success looks like: intelligible Swahili-English with the speaker's timbre, RTF
well above 1 on an L4. Failure looks like: `rms < 0.001` (silent - broken
inference), `rms > 0.5` with a short duration (noise - LR too high or too many
steps), or audio that never ends (EOS forgotten - same cause).

The dataset transcripts are verbatim monologue, and every clip is exactly 30 s
because it was cut on a grid, so some cuts land mid-word. Leave the transcripts
alone: rewriting them is what misaligns text and audio.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `RuntimeError: The size of tensor a (2048) must match tensor b (1024)` | you are running the *upstream* `sft_12hz.py`; use `sheng-qwentts train` |
| `AutoModel returned <other class>` / processor errors | `pip install -U qwen-tts` (needs `transformers==4.57.3`) |
| `assert sr == 24000` | wavs did not come from stage `data`, or `ref_audio` is not 24 kHz |
| `FileNotFoundError: .../model.safetensors` | run `sheng-qwentts model` (a repo id will not work) |
| CUDA OOM on T4 | `--profile t4 --allow-small-gpu --steps 240`; keep `batch_size*grad_accum = 4` |
| training loss falls, output is noise | halve `--lr` (1e-6) and/or stop earlier |
| Drive runs out of space | `--keep-last 2`, delete old `evals/` |

## Repo layout

```
src/sheng_qwentts/
  cli.py        the driver (status/setup/data/model/codes/train/eval/export/all)
  data.py       parquet -> 24 kHz wavs + manifests + report
  resample.py   22.05k -> 24k polyphase (numpy fallback, scipy when present)
  model.py      snapshots + Qwen3TTSModel loading (sdpa fallback)
  codes.py      12 Hz tokenizer -> audio_codes
  batching.py   the interleaved text/codec matrix (portfolio of the upstream collate)
  embeds.py     text_projection fix + speaker slot (the 0.6B bug lives here)
  train.py      the SFT loop (argparse only, no accelerate)
  infer.py      generate_custom_voice / zero-shot A/B / export
  progress.py   Drive-persisted stage state + progress.md
  env.py        environment checks, GPU -> profile
tests/          66 tests, no GPU and no network needed
notebooks/      qwentts_sheng_colab.ipynb
```

## Run the tests

```bash
pip install -e ".[dev]"
pytest -q          # 66 passed
```

## Honest limits

* 21 minutes of audio is a **voice** fine-tune. Do not expect Sheng slang the
  base model has never seen to be pronounced correctly; keep prompts in the
  dataset's register (Swahili orthography + English words).
* single speaker only - multi-speaker SFT is not in the upstream release.
* the checkpoint drops the speaker encoder (by design: the voice is baked into
  `codec_embedding[3000]`), so it cannot be re-cloned from new audio.
* licence: the model is Apache-2.0, but check the rights of the source video the
  dataset was cut from before publishing an exported voice.
