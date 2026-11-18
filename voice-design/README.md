# Kenyan voice design on VoxCPM2 - Colab LoRA

Teach [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) (2B, Apache-2.0) to
**design Kenyan-sounding voices from a text description**.

VoxCPM2 already has a *voice design* mode: a description in parentheses in front of the
text conditions the synthesised voice, no reference audio needed.

```python
model.generate(text="(A Kenyan female voice in her thirties, mid-pitched)Habari yako?")
```

Out of the box that space is mostly English-accented. This repo fine-tunes a LoRA on
Kenyan speech where **every transcript is prefixed with a truthful description of that
speaker's voice**, so the caption ↔ voice mapping moves to Kenyan voices - in Swahili and
in the English words Kenyans mix in.

```
notebooks/          the Colab notebook (start here)
configs/            readable reference YAML per GPU profile
scripts/            colab_bootstrap.sh, render_configs.py
src/kenya_voices/   the pipeline (data, captions, config, train, eval, export, cli)
app/                small Gradio playground for the trained adapter
tests/              99 tests, no GPU / torch / network needed
progress.md         live status journal - survives a disconnect
```

## Quick start (Colab)

1. Runtime > Change runtime type > **L4** (recommended) or A100.
2. Open `notebooks/kenya_voice_design_colab.ipynb` and **Run all**.

Or by hand:

```bash
%cd /content/audio-speech-ai/voice-design
!pip install -q -e .
!kenya-voices doctor                    # GPU / Drive / deps / model / data checklist
!kenya-voices all --resume --max-hours 8
```

`all` runs `env -> data -> model -> train -> eval -> export` and **skips stages that are
already done**, so the same command is the recovery command after a disconnect.

Step by step:

```bash
!kenya-voices env                 # clone + install OpenBMB/VoxCPM (--no-deps + curated deps)
!kenya-voices data                # Kenyan voices -> 16 kHz WAVs -> caption manifests
!kenya-voices model               # snapshot openbmb/VoxCPM2 to a local dir
!kenya-voices train --max-hours 8 # caption-conditioned LoRA, resumes automatically
!kenya-voices eval --ab           # design prompts, caption adherence, LoRA vs base
!kenya-voices export              # portable adapter + design palette + samples
!kenya-voices design --caption "A Kenyan male voice in his thirties, low-pitched" \
                     --text "Habari yako?" --lora <export dir>
```

## How the "design a voice" part works

Three decisions carry the whole result:

**1. Captions in the text, not a new model head.** Upstream's trainer tokenises each
row's `text` as the generation target. Putting `"(A Kenyan woman in her thirties,
mid-pitched)Habari yako"` in `text` trains exactly the same conditioning that inference
uses - no code changes in the model, and the LoRA learns to bind description to voice.

**2. Captions only describe what is true.** `src/kenya_voices/captions.py` builds each
description from:

| Attribute | Where it comes from |
|---|---|
| gender | dataset metadata (`gender`) |
| age | dataset metadata (`age_group`) |
| pitch | **measured** median f0 per speaker (`audio_features.median_f0`) |
| pace | **measured** words per second |
| place | dataset metadata (`location`, e.g. `from Nairobi`) |
| language mix | **measured** English word share (Swahili vs. Swahili + English / Sheng) |

Invented adjectives ("warm", "authoritative") are unlearnable labels that just add noise;
they make the adapter ignore the caption. Attributes are aggregated **per speaker**, so
all of one speaker's clips carry the same description - a design attribute belongs to a
voice, not to a single clip.

**3. The data is set up so the caption has to generalise.** Validation speakers are held
out entirely (speaker-disjoint split), and ~10% of rows keep a plain transcript so the
adapter still works without a description.

## Data

Default preset: [`badrex/swahili-speech-400hr`](https://huggingface.co/datasets/badrex/swahili-speech-400hr)
- 64.5k clips, ~380 Kenyan speakers, 16 kHz, CC-BY-4.0, with gender / age / location
metadata. `kenya-voices datasets` also lists a smaller custom preset
(`lyle-mlengineer/kenyan-voices`, audio + text only) and shows the column mapping a
preset needs - that is how you plug in **your own** dataset.

The source is ~44 GB, so nothing is downloaded wholesale:

1. **Metadata pass** (`--columns` without audio): per-speaker clip counts, durations,
   gender/age/location votes. Cheap.
2. **Speaker selection**: hash order (deterministic), gender-mixed, capped by
   `--speakers` (120) and `--hours` (10 h).
3. **Audio pass**: a filtered stream (`input_columns=["audio_id"]`) decodes only the
   selected clips, resamples to 16 kHz mono PCM16, measures f0, writes WAVs to Drive.

Defaults: 120 speakers x 30 clips, ~10 h. That is enough to move the design space
(Kenyan accent, pitch/age/place realisations) without a multi-day run. WAVs and manifests
are reused on rerun, so a disconnect never re-streams the dataset.

## Training

Rendered by `src/kenya_voices/config.py` (templates in `configs/` are kept in sync by
`tests/test_config_templates.py`):

- LoRA `r=32`, alpha 32 on **LM + DiT**, `enable_proj: false` - ~20 GB VRAM, so L4 (24 GB)
  or A100. T4 only via `--profile t4 --allow-small-gpu` (no native bf16).
- Effective batch 16 on every profile (`2x8` L4, `4x4` A100, `1x16` T4).
- `sample_rate: 16000` is mandatory - it is asserted against `audio_vae.sample_rate`
  (AudioVAE encoder input). `out_sample_rate: 48000` is the decoder and only affects
  TensorBoard audio. Never edit it.
- `num_iters == max_steps` so "steps" means one thing; default 1200 (~2 epochs).
- Checkpoints `step_%07d` + `latest/` on Drive, `save_interval 100`.
- `--max-hours` interrupts gracefully; the trainer checkpoints on SIGINT.

If the adapter turns out to ignore descriptions (caption adherence ≈ base model), the two
cheapest fixes are `--caption-variants 2` (same clips, two phrasings - forces the model to
read the caption rather than memorise one sentence) and more speakers rather than more
steps.

## Judging the adapter

`kenya-voices eval` renders seven fixed design prompts and writes `report.md` with an
objective proxy next to the wavs:

- **caption adherence** - a "male, low-pitched" caption must produce a lower median f0
  than "female, higher-pitched" *on the same text*. Score per variant (LoRA vs base with
  `--ab`). If the LoRA does not beat the base model here, the caption signal was not
  learned.
- **text adherence** - `unseen_vocab` and `long_form_narrator` must say new words; a
  checkpoint that replays training audio has overfit (the usual failure).
- **Kenyan-ness** - ears only: the accent must show up in English words too, not just in
  Swahili.

Pick the **earliest** checkpoint that sounds right, then `export` it.

## Serving the adapter

```python
from voxcpm.core import VoxCPM

model = VoxCPM.from_pretrained(
    hf_model_id="openbmb/VoxCPM2",
    load_denoiser=False,
    lora_weights_path="exports/design_r32/step_0001200",  # contains lora_config.json
)
audio = model.generate(
    text="(A Kenyan female voice in her thirties, mid-pitched, speaking Swahili)Habari yako?",
    seed=42,
)
```

Exports rewrite `base_model` to the Hub id, so the folder is portable. `design_prompts.json`
inside carries captions the adapter was trained on - a ready-made design palette.

A thin Gradio front-end (caption presets, text, seed, CFG):

```bash
python app/voice_design_app.py --lora artifacts/exports/design_r32/step_0001200
```

## Where things live

| What | Path |
|---|---|
| artifacts root | `/content/drive/MyDrive/kenya-voices` when Drive is mounted, else `<repo>/artifacts` |
| 16 kHz wavs / manifests | `<artifacts>/data/wav16k`, `<artifacts>/manifests` |
| speaker profiles / val speakers | `<artifacts>/data/meta/speakers.json`, `val_speakers.json` |
| rendered train YAML | `<artifacts>/configs/<run>.yaml` |
| checkpoints | `<artifacts>/checkpoints/<run>/step_%07d` + `latest/` |
| logs | `<artifacts>/logs/<run>/console.log`, `<artifacts>/checkpoints/<run>/train.log` |
| evals / exports | `<artifacts>/evals/<run>`, `<artifacts>/exports/<run>` |
| progress journal | `progress.md` (repo root + artifacts), `<artifacts>/progress/state.json` |

Override the root with `--artifacts PATH` or `$KENYA_ARTIFACTS`.

## Tests

```bash
pip install -q pytest
python -m pytest -q          # 99 tests, no GPU, no torch, no network
```

They cover the caption contract (truthful attributes, varied phrasings, caption
round-trip), the pitch tracker, the JSONL manifest contract, speaker-disjoint splitting,
caption dropout/variants, the rendered YAML invariants, template sync, checkpoint
discovery/resume, the progress journal, and the CLI.

## Limits to be honest about

- **10 h / 120 speakers is an accent-and-timbre adaptation, not a new language.** VoxCPM2
  already speaks Swahili; the LoRA re-centres *which voices* the design space produces.
- **Out-of-distribution captions are a gamble.** Emotion or style words were never labels,
  so they may leak from the base model but are not guaranteed.
- **Overfitting is the main failure mode.** More steps after the val loss turns is almost
  always worse.
- **One adapter cannot encode every Kenyan voice** - it shifts the distribution. For a
  specific person, use VoxCPM2's cloning modes instead.
- **Licence**: dataset CC-BY-4.0 (attribute `badrex/swahili-speech-400hr`), base model
  Apache-2.0. Check the licence of any preset you add - several Kenyan speech sets are
  CC-BY-NC (non-commercial).

## Next step

Run `notebooks/kenya_voice_design_colab.ipynb` cells 0-6 on an L4 (that is `doctor` +
`data` + `all --skip-eval --skip-export`), then listen to the first checkpoint before
spending more GPU hours.
