# Sheng Whisper Fine-Tuning — Progress Log

> **Purpose:** survive a Colab disconnect. If the runtime dies, read the last
> entry here, then follow *Resume procedure* below. Keep this file updated as
> you go — append a new entry after every meaningful run.

---

## Status

| Field | Value |
| --- | --- |
| Current phase | **Code complete, not yet trained** |
| Last update | 2026-11-15 |
| Base model | `openai/whisper-medium` |
| Dataset | `lyle-mlengineer/kenyan-celebs` |
| Best WER so far | _none yet_ |
| HF repo | `lyle-mlengineer/whisper-medium-sheng` (not pushed yet) |
| GPU used | Colab T4 (15 GB), not yet run |

---

## What is built (verified locally)

- [x] `src/whisper_sheng/config.py` — typed YAML config, model size swappable,
      unknown keys rejected, CLI `--set section.key=value` overrides.
- [x] `src/whisper_sheng/text_norm.py` — Sheng normalisation (lowercase,
      punctuation strip, elongation collapse `nomaaa`→`nomaa`).
- [x] `src/whisper_sheng/metrics.py` — WER, CER, mixed-utterance WER,
      code-switch rate error, per-example report.
- [x] `src/whisper_sheng/data.py` — load, split, 16 kHz cast, speed perturb,
      SpecAugment.
- [x] `src/whisper_sheng/tracking.py` — MLflow init, dataset stats, JSON
      progress sidecar.
- [x] `src/whisper_sheng/trainer.py` — model load, collator, training args
      (handles transformers 4.x **and** 5.x), final eval, bundle save.
- [x] `scripts/train.py` — orchestration, auto-resume, HF push, model card.
- [x] `configs/whisper_medium_t4.yaml` — T4 defaults for whisper-medium.
- [x] `configs/whisper_small_t4.yaml` — fast iteration variant.
- [x] `notebooks/colab_finetune.ipynb` — 22 cells, valid JSON.
- [x] `tests/test_core.py` — **27 tests, all passing** (no GPU needed).

**Locally verified:** config loading + overrides, text normalisation on real
dataset rows, all metrics, augmentation determinism. **Not verified:** an
actual training run (needs a GPU) and the HF push (needs a token).

---

## Dataset facts (verified via HF API, 2026-11-15)

These are facts, not assumptions — they were checked against the live dataset:

| Property | Value |
| --- | --- |
| Rows | **42** |
| Splits | **`train` only** (no validation/test split) |
| Columns | `audio` (16 kHz), `speaker`, `text` |
| Speakers | **1 — all rows are `abel`** |
| Clip length | **30.0 s for every row** |
| Transcript length | 289–501 chars (mean ~415) |
| Timestamps | none |
| Language tags | none |

### Why this matters (read before trusting any metric)

1. **42 rows is far below what Whisper fine-tuning wants.** Published Whisper
   fine-tunes use tens of hours. This is ~21 minutes. The model will overfit.
   Expect training loss to fall far below eval loss.
2. **One speaker ⇒ eval WER is not a generalisation estimate.** A held-out
   split is still the *same person*, so a good WER partly measures "did it
   memorise this voice". Treat WER as a *regression signal*, never as a claim
   about Sheng in general.
3. **No `sheng` token exists in Whisper.** We pin language to `sw`. Only the
   decoder prompt is constrained, so English words still come out.
4. **30 s clips sit exactly on Whisper's input limit**, so there is no chunking
   logic — and no context beyond a single clip.

If you want a number you can actually publish, the honest path is more Sheng
audio from more speakers (e.g. `lyle-mlengineer/kenyan-voices` has 75 more rows,
still too few, but it is a better *test* set because it is a different crawl).

---

## Resume procedure after a disconnect

**1. Reopen the notebook**, set runtime to T4, and re-run cells 1–4
(GPU check, clone, credentials, Drive mount, deps). These are all fast.

**2. Check what survived on Drive:**

```python
!ls -1 /content/drive/MyDrive/whisper-sheng/whisper-medium-sheng/
!cat /content/drive/MyDrive/whisper-sheng/whisper-medium-sheng/trainer_progress.json
```

`trainer_progress.json` holds `last_step`, `last_loss`, `eval_history`, and the
`mlflow_run_id`.

**3. Re-run cell 7 (full fine-tune).** `scripts/train.py` automatically finds
the newest `checkpoint-*` in `output_dir` and resumes from it. There is no
special flag needed — just re-run.

**4. To resume a specific checkpoint instead:**

```python
!python scripts/train.py --config configs/whisper_medium_t4.yaml \
    --resume /content/drive/MyDrive/whisper-sheng/whisper-medium-sheng/checkpoint-120 \
    --set train.output_dir={DRIVE_DIR}/whisper-medium-sheng \
    --set mlflow.tracking_uri=sqlite:///{DRIVE_DIR}/mlflow.db
```

**5. To re-attach to the previous MLflow run** (rather than starting a new one),
add to the resume command: `--set mlflow.run_name=<original name>`. MLflow will
create a new run with the same name; the original `run_id` is in
`trainer_progress.json` if you need to merge history by hand.

### If Drive was *not* mounted

Then everything is gone. Colab does not persist `/content`. Re-run from
scratch, and next time mount Drive **before** training (cell 9 does this).

---

## Expected cost / timing on a T4

Rough estimates, to be replaced with real numbers after the first run:

| Stage | Estimate |
| --- | --- |
| pip install | 2–4 min |
| Dataset download (52 MB) | < 1 min |
| Preprocessing + augmentation | 1–2 min |
| Training, whisper-medium, 12 epochs | **~1.5–2.5 h** |
| Final eval (37 rows × generate) | 1–3 min |
| HF push | 1–2 min |

**Colab free tier disconnects after ~4 h of inactivity and can drop without
warning.** A 2 h run is plausible but not guaranteed. Checkpoints every 20
steps (~every 4 min) mean worst case you lose ~4 minutes.

---

## Log

Append new entries at the top of this section (newest first).

### 2026-11-15 — Code complete

- Built the full pipeline; 27/27 unit tests pass locally (Python 3.12).
- Fixed a real bug found by tests: `spec_augment` filled masks with the
  array's *global* mean, which on near-constant input was a literal no-op.
  Now bounds-checked and validated with varied features, plus added
  determinism and narrow-input tests.
- Fixed notebook JSON (was malformed on first write); validated all 11 code
  cells parse as IPython.
- **Next:** run smoke test on Colab T4, then the full run.

### Template for new entries

```
### YYYY-MM-DD — <what ran>

- Config: <config file> / overrides used
- Command: <exact command>
- Result: WER=…, CER=…, mixedWER=…
- Steps completed: … / …  (epoch …)
- Notes: <OOM? slowdown? weird predictions?>
- Next: <concrete next action>
```

---

## Known risks & open decisions

| Risk | Mitigation / status |
| --- | --- |
| **Overfitting on 42 rows** | Augmentation + early stopping + `load_best_model_at_end`. Accept that best WER is likely on an early epoch. |
| **Eval set is ~6 rows** | Metrics are very noisy. Do not chase 0.01 WER differences. `mixed_wer` may be `nan` if no eval row has English. |
| **8-bit Adam unavailable** | Config falls back to fp32 AdamW, which may OOM medium on a T4. If so, lower `per_device_train_batch_size` to 2. |
| **`fp16` NaN loss** | T4 + fp16 occasionally produces inf/nan on Whisper. If loss goes `nan`, set `--set train.fp16=false` (slower, bigger memory). |
| **HF push permissions** | Token needs **write** scope. Verified manually in the notebook (cell 19). |
| **Whisper 5.x API drift** | `trainer.py` introspects `Seq2SeqTrainingArguments` and picks `eval_strategy` vs `evaluation_strategy`, and passes `processing_class`. Both 4.x and 5.x work. |

## Next actions

1. **Smoke test on T4** (cell 13) — validates deps, OOM, label padding. ~2 min.
2. **Full whisper-medium run** (cell 15) — expect a mediocre WER; that is the
   honest result for 42 rows.
3. **Read predictions** (cell 17) — check whether Sheng slang survives. This is
   more informative than the WER number.
4. Optional: train `whisper-small` via `configs/whisper_small_t4.yaml` to see if
   the smaller model generalises better under this data constraint.
