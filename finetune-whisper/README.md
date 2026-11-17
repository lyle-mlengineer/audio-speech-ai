# Fine-tune Whisper for Sheng (Kenyan Swahili-English)

Fine-tunes a **configurable** Whisper checkpoint (`whisper-medium` by default) on
Sheng, the Swahili/English code-switched variety spoken in Kenya, using Google
Colab's free T4. Tracks runs in **MLflow** and pushes the result to the
**Hugging Face Hub**.

## Quick start (Colab, T4)

Open `notebooks/colab_finetune.ipynb` in Colab, set **Runtime → T4 GPU**, add an
`HF_TOKEN` secret, and run top to bottom. The notebook smoke-tests in ~2 minutes
before committing to the full run.

```bash
# local equivalent
python scripts/train.py --config configs/whisper_medium_t4.yaml --smoke-test --no-push
python scripts/train.py --config configs/whisper_medium_t4.yaml
```

## Switching model size

The model is just a config value — no code changes:

```bash
python scripts/train.py --config configs/whisper_medium_t4.yaml \
    --set model.name=openai/whisper-large-v3
```

Any HF Whisper checkpoint works. Batch sizes in the YAML are tuned for a 15 GB
T4; going larger than `medium` will need a smaller batch and likely an L4/A100.

## Layout

```
finetune-whisper/
├── configs/
│   ├── whisper_medium_t4.yaml   # T4 defaults, whisper-medium
│   └── whisper_small_t4.yaml    # fast iteration
├── notebooks/colab_finetune.ipynb
├── scripts/train.py             # entrypoint
├── src/whisper_sheng/
│   ├── config.py                # typed YAML config + overrides + validation
│   ├── data.py                  # load, split, augment, feature prep
│   ├── metrics.py               # WER / CER / mixed-WER / code-switch rate
│   ├── text_norm.py             # Sheng-aware normalisation
│   ├── tracking.py              # MLflow + JSON progress sidecar
│   └── trainer.py               # model, collator, args, eval, save
├── tests/test_core.py           # 27 tests, no GPU needed
└── progress.md                  # resume log — read this after a disconnect
```

## Design notes

**Language pinning.** Whisper has no `sheng` language token. We pin the decoder
prompt to `sw`/`transcribe`. Because only the *prompt* is constrained, not the
output vocabulary, English insertions still appear in the transcript. This is the
practical way to handle an unsupported mixed language.

**Text normalisation is applied twice on purpose** — once when building training
labels and once when scoring. Sheng has no standard orthography, so without it,
WER measures spelling inconsistency rather than recognition quality. Elongation
is capped (`nomaaa` → `nomaa`) since 42 examples cannot teach 4-way length
distinctions.

**Metrics beyond WER.** `mixed_wer` restricts scoring to utterances that
actually contain English, so a model that handles Swahili but drops code-switched
tokens cannot hide inside an aggregate average.

**Resilience.** MLflow writes to a SQLite DB (point it at Drive), checkpoints
land every 20 steps, and `scripts/train.py` auto-resumes from the newest
`checkpoint-*`. A JSON sidecar (`trainer_progress.json`) records the step, loss
history and MLflow `run_id` independent of MLflow.

**transformers 4.x / 5.x compatibility.** Colab now ships transformers 5.x, where
`evaluation_strategy` became `eval_strategy` and `tokenizer=` became
`processing_class=`. `trainer.py` introspects the signature and passes whichever
the installed version accepts, so the same code runs on local 4.x and Colab 5.x.

## Important caveat about the dataset

`lyle-mlengineer/kenyan-celebs` has **42 rows, one speaker, and a single `train`
split**. This is roughly 21 minutes of audio for a 769M-parameter model, so:

- Expect overfitting; the best checkpoint is likely an early epoch.
- The held-out WER is measured on the **same speaker**, so it is a regression
  signal, not a claim about generalising to new Sheng speakers.
- Do not publish the WER as if it were a general Sheng benchmark.

`progress.md` documents the dataset facts in full, plus what an honest next step
would be.

## Tests

```bash
python -m pytest tests/ -q          # or: python tests/test_core.py
```

Covers config validation and overrides, normalisation on real Sheng strings,
all metrics including edge cases (empty hypotheses, single-class splits), and
augmentation determinism.
