# How Chatterbox Was Trained — Upstream Pretraining and This Project's Fine-Tune

Two different trainings get confused with each other:

1. **Upstream pretraining** of `ResembleAI/chatterbox` — 23 languages, the
   0.5 B T3 LM + S3Gen vocoder + voice encoder. Resemble AI's data and compute.
   No training script is published, so this part is reconstructed from the
   *shipped weights, configs and inference code*.
2. **The fine-tune in this repository** — `finetune-chatterbox` (`sheng_t3`),
   which trains only T3 on 21 minutes of Sheng. Everything here is verifiable
   line by line.

Claims below are tagged: **[verified]** = read from shipped code/artifacts,
**[inferred]** = a design deduction from how the code behaves, **[unknown]** =
not determinable from the public repository. The tags matter: a training recipe
built on guessed hyperparameters wastes GPU hours.

---

# Part I — How the released model was (probably) trained

## 1.1 What is actually observable

| artifact | what it pins down |
|---|---|
| `t3_mtl23ls_v2.safetensors` | 30-layer Llama, hidden 1024, 16 heads x 64, MLP 4096, RoPE theta 5e5 with llama3 scaling x8; custom `text_emb` 2454, `speech_emb` 8194, learned absolute pos tables, two linear heads |
| `grapheme_mtl_merged_expanded_v1.json` | 2454-id character-level BPE; 23 `[xx]` language tags; **IPA phone token set** (`ˈiː`, `tʃ`, `ɪŋ`, ...); **non-verbal tags** `[laughter] [sigh] [inhale] [gasp] [whisper] [guffaw] [cough] [shhh] [gibberish] ...` (118 added tokens) |
| `s3gen.pt` | S3 tokenizer (`speech_tokenizer_v2_25hz`, 6561 codes), CAMPPlus x-vector, 6-block UpsampleConformer, `ConditionalDecoder` (320->80, 12 mid blocks), HiFT with `[8,5,3]` |
| `ve.pt` | Resemblyzer-lineage GE2E encoder, 256-d, 16 kHz |
| `conds.pt` | one shipped `Conditionals` object (T3Cond + s3gen ref_dict) — a built-in voice |
| code | `T3.loss` (dual-stream CE), `CFM.compute_loss` (flow-matching MSE + 20 % conditioning dropout), `inference_cfg_rate = 0.7`, `AlignmentStreamAnalyzer` head indices |

Three of those are unusually informative:

* **IPA tokens in the text vocabulary** mean the upstream data pipeline had a
  phoneme mode (`[ipa]` tag). Grapheme and phoneme training for the same
  utterance is a cheap way to teach pronunciation instead of spelling.
* **Non-verbal and singing tags** mean the training audio was *not* clean
  single-speaker studio TTS: those tags exist because the data contained
  laughter, sighs, breaths and music. It is in-the-wild speech.
* **`AlignmentStreamAnalyzer` hard-codes layers 12/13/9 and heads 15/11/2.** Those
  heads were found by *inspecting a trained model* and seeing which ones encode
  monotonic text-speech alignment. So alignment is **emergent**, not supervised by
  an auxiliary loss in the released code. Any recipe that adds an explicit
  duration/alignment loss is a departure from the original.

## 1.2 The data pipeline, reconstructed

```
raw audio (16 kHz mono)                          raw transcripts
        │                                                │
        │  S3TokenizerV2 @ 25 Hz                         │  MTLTokenizer
        ▼                                                ▼
speech tokens, 6561-code alphabet                grapheme ids + [lang] tag
(40 ms per token, 2 mel frames)                  (optional [ipa] phone ids)
        │                                                │
        │  ve.embeds_from_wavs (256-d, mean over windows) │
        ▼                                                │
speaker embedding, 1 per utterance                       │
        │                                                │
        └──────────────► training example ◄───────────────┘
             cond_prompt_speech_tokens = first 150 S3 tokens of the clip
             (or of a same-speaker clip — the "in-context voice prompt")
```

Design consequences that are **[verified]** by the shapes:

* The speech tokenizer is fixed and *offline*: T3 is trained on discrete ids, so
  the target is a classification problem over 6561 classes — no mel-spectrogram
  regression, no duration model, no variance adaptor.
* The reference prompt is exactly `speech_cond_prompt_len = 150` tokens = **6 s**,
  matching `ENC_COND_LEN = 6 * S3_SR` at inference. Training and inference see the
  same conditioning budget.
* `emotion_adv` is a *scalar* projected to 1024-d by a bias-free linear layer.
  It is trained as a control knob; the shipped default 0.5 is the neutral point,
  which is why `finetune-chatterbox` must keep `exaggeration=0.5` — a fine-tune
  sampled at a different value teaches a different voice.
* `max_text_tokens = 2048` and `max_speech_tokens = 4096` imply training clips up
  to ~2-3 minutes of text and ~160 s of speech; the learned position tables are
  sized `2048+2` and `4096+4`.

## 1.3 The T3 objective

The only published loss is `T3.loss` **[verified]**:

```python
hidden = LlamaModel(inputs_embeds = cat([cond(34), text_emb(L+2), speech_emb(S+2)]))
text_latents   = hidden[:, 34       : 34+L+2]     -> text_head   -> CE vs text_tokens
speech_latents = hidden[:, 34+L+2   : 34+L+2+S+2] -> speech_head -> CE vs speech_tokens
loss = loss_text + loss_speech
```

Structurally this is a **two-stream next-token model over one shared causal
trunk**, where:

* the *speech* stream is the real objective — it is what gets sampled at
  inference;
* the *text* stream is an auxiliary reconstruction task that keeps the
  transcript recoverable from the hidden states. Without it, a pure
  speech-token LM can drift away from the text and start babbling; with it, the
  state at speech position `p` still contains "what word am I in".

**[inferred]** The upstream trainer almost certainly used the **shifted** form
(position `p` predicts token `p+1`) and masked the prompt region out of the
speech labels, because that is what generation actually needs and what the
independent community kit implements. The shipped `T3.loss` is unshifted, which
only makes sense as a helper for a caller that passes pre-shifted tensors. See
§2.6 — this is the single most important detail to get right when fine-tuning.

**[inferred]** The text stream must have been trained with *text dropout*
(occasionally replacing the text embeddings with zeros), because at inference
`prepare_input_embeds` does exactly that to build the unconditional CFG branch:

```python
if cfg_weight > 0.0:
    text_emb[1].zero_()   # CFG uncond
```

Guidance with an un-trained zero branch would produce nonsense, and CFG
(`cfg_weight=0.5`) is a shipped default, so a zeroed-text condition was almost
certainly in the training distribution. The same argument applies to speaker /
prompt dropout, which the community kit does explicitly at 20 %.

## 1.4 The s3gen objective

`CausalConditionalCFM.compute_loss` **[verified]**:

```python
t  = rand();  if cosine: t = 1 - cos(t * pi / 2)
z  = randn_like(x1)                                  # noise
y  = (1 - (1 - sigma_min) * t) * z + t * x1          # interpolate noise -> data
u  = x1 - (1 - sigma_min) * z                        # the true velocity
if training_cfg_rate > 0:                            # 20 %
    mu, spks, cond = mu*cfg_mask, spks*cfg_mask, cond*cfg_mask
pred = estimator(y, mask, mu, t, spks, cond)
loss = mse(pred * mask, u * mask) / (sum(mask) * 80)
```

So s3gen is **flow matching with a linear (rectified-flow) path** and a cosine
`t` sampler, MSE on the velocity field, with **20 % joint conditioning dropout**
(content, speaker, reference mel all dropped together). That dropout is exactly
what makes the speaker-only CFG at inference (0.7) meaningful.

The HiFT vocoder's adversarial/mel losses are **not** in the package
**[unknown]**; the architecture (`[8,5,3]`, 3 fused ResBlocks, F0-driven harmonic
source, `istft n_fft 16 / hop 4`) is standard HiFTNet, whose published recipe is
LSGAN + mel L1 + feature matching with an F0 auxiliary term. Treat that as the
prior, not as a fact about these weights.

`token_mel_ratio = 2` and the `mel_len == 2 * token_len` assertion in `embed_ref`
are the contract that keeps CFM training frames aligned to token boundaries: 25
tokens/s x 2 = 50 mel frames/s (`hop 480` at 24 kHz).

## 1.5 Order of training and why it matters

**[inferred]** The dependency graph forces a staged build:

```
1. vehicle:  ve (GE2E speaker encoder)      needs only speaker labels
2. tokens:   S3 tokenizer (VQ/FSQ over mel) needs only audio
3. vocoder:  s3gen = CFM(tokens|text-enc, speaker, ref-mel) + HiFT(mel -> wav)
4. LM:       T3 = (text, speaker, prompt tokens) -> speech tokens
```

T3 can only be trained after 1-3 exist, because its targets *are* S3 tokens and
its conditioning *is* a `ve` embedding. This is also why the released weights are
four files and why fine-tuning T3 alone is legitimate: T3 sits at the top of a
frozen pyramid.

A useful sanity check on this ordering: a T3 trained against a *different*
tokenizer or a *different* `ve` would be silently incompatible. `voice-clone`'s
`load_engine` explicitly counts missing/unexpected keys and refuses if more than
10 % of the keys do not fit, precisely because swapping a mismatched T3 produces
fluent-looking garbage.

## 1.6 What we do not know

| question | status |
|---|---|
| total hours of audio | **unknown** (nothing in the repo; the 23-language scope implies thousands of hours, but any number would be invented) |
| per-language balance | **unknown** |
| steps / batch / LR / hardware | **unknown** |
| whether transcripts came from ASR or human labels | **unknown**; the tag set (laughter, cough, music) suggests ASR + heuristics |
| text dropout rate for CFG | **unknown**, but > 0 is implied by the shipped `cfg_weight` path |
| T3 learning-rate schedule | **unknown** |
| how aligned heads 12/13/9 were chosen | **unknown**; the code calls them `LLAMA_ALIGNED_HEADS` with no derivation |

If you need a defensible number for any of these, the only honest approach is to
measure it yourself with an ablation (§3.3).
---

# Part II — The fine-tune in this repository (`sheng_t3`)

## 2.1 The decision that defines the recipe

```
trainable:    t3   (~541 M params incl. both embedding tables)
frozen:       ve, s3gen.tokenizer, s3gen.flow (CFM), s3gen.mel2wav (HiFT)
data:         38 clips x ~30 s = ~19 min, one speaker ("abel"), Sheng
objective:    CE(text) + CE(speech) on the frozen tokenizers
hardware:     Colab L4 24 GB (default), A100, or T4 (fp16 only)
wall clock:   ~1.5-3 h for 600 steps on an L4
```

Why T3 only: timbre, accent and code-switch prosody live in the token sequence.
The vocoder is 200 M parameters of audio fidelity that 19 minutes of data cannot
improve and can easily damage. Freezing it makes "audio quality regression"
impossible by construction — the failure mode becomes "wrong tokens", which is
audible and measurable.

Why the multilingual checkpoint: `sw` is already in `SUPPORTED_LANGUAGES`, so the
fine-tune learns *Sheng delivery* (Swahili matrix + English content words + one
voice), not Swahili phonology from scratch. Starting from the English checkpoint
(`--model-variant english`) would have to learn Swahili *and* the voice from 19
minutes, which is not a realistic ask.

## 2.2 Data stage

`dataset.py` **[verified]**:

```
lyle-mlengineer/kenyan-celebs  (42 clips, speaker "abel", ~30 s each, 22.05 kHz)
  -> mono, resample to 16 kHz, PCM16 WAV        data/wav16k/abel_00NN.wav
  -> shuffle(seed=1337), val = 15 % (min 1, capped by --val-clips, default 4)
  -> manifests/train.jsonl, manifests/val.jsonl  {"audio","text","seconds","speaker","language"}
```

Deliberate choices:

* **Clips are kept whole.** Splitting a 30 s clip needs forced alignment to keep
  transcripts word-accurate; a wrong transcript hurts cloning more than a long
  clip does.
* **No text normalisation beyond whitespace.** Sheng orthography is what the
  speaker actually said; "correcting" it teaches the model the wrong mapping.
* **16 kHz, not 24 kHz.** Both consumers of the input audio (`ve`, S3 tokenizer)
  are 16 kHz; only the *output* is 24 kHz, and s3gen is untouched.
* A code-switch marker statistic (`code_switch_stats`) is computed as a sanity
  signal — it is a rough English/Swahili word-share, not a language ID.

## 2.3 Featurisation

`t3data.py::Featurizer` builds exactly what `generate` consumes at inference:

| field | source | shape |
|---|---|---|
| `text_tokens` | `MTLTokenizer(text, language_id="sw")` + `[START]`/`[STOP]` | `(L+2,)` |
| `speech_tokens` | `s3gen.tokenizer(wav16k)` + start/stop speech ids | `(S+2,)` |
| `t3_cond_speaker_emb` | `ve.embeds_from_wavs([wav16k])`, reshaped to 256 | `(256,)` |
| `t3_cond_prompt_speech_tokens` | S3 tokens of the **first 6 s**, zero-padded/truncated to 150 | `(150,)` |
| `t3_cond_emotion_adv` | `0.5` | scalar |

Cost control **[verified]**: featurisation runs a full mel + quantiser forward per
clip and is cached as `.npz`, keyed by
`sha1(audio_path | mtime | transcript | featuriser signature)`. Editing a
transcript or changing `cond_seconds` invalidates the entry automatically; a
corrupt cache entry is treated as a miss. On a Colab disconnect you therefore
re-enter training with no featurisation cost.

The important subtlety: **the prompt tokens are taken from the same clip that is
being predicted.** That is teacher forcing on the conditioning: the model sees the
first 6 s of the target and must produce the rest. At inference the prompt comes
from a *different* recording of the same speaker. Fine-tuning long enough on
"prompt == prefix of target" is one mechanism behind the well-known degradation
where the model starts reciting training clips — prompt and target become too
correlated.

## 2.4 Batching

`SpeechCollator` **[verified]**:

```
text_tokens      (B, max_text)     pad 0
speech_tokens    (B, max_speech)   pad 0
text_token_lens  (B,)
speech_token_lens(B,)
speaker_emb      (B, 256)
prompt_tokens    (B, 150)          fixed length, no padding
emotion_adv      (B, 1, 1)
```

Two things to know:

* `T3.loss` asserts `len_text == text_token_lens.max()`, so a batch is padded to
  its own maximum and the mask is built from the lens. No cross-batch waste.
* `max_speech_len` defaults to 1024 = ~41 s at 25 Hz, so 30 s clips always fit;
  `max_text_len` 512 is far above the ~450-character transcripts.

`T3Dataset.__getitem__` returns `None` on a bad clip and records it in
`self.failures`; the collator drops `None`s and the trainer aborts only after 50
empty batches in a row. One corrupt WAV cannot kill a run.

## 2.5 Optimisation

**[verified]** defaults (`train_t3.py`, `config.py::PROFILES`):

| item | value | why |
|---|---|---|
| optimiser | `AdamW(betas=(0.9, 0.95), wd=0.01)` | LLM-standard; β2 below 0.999 suits short runs |
| LR | `5e-5` | full fine-tune of a 0.5 B LM on 19 min; this model family is reported to produce noise / endless generation above ~2e-5 for long runs |
| schedule | linear warmup 50 steps, then cosine to 0 | a pure function of `step`, so resume is exact |
| steps | 600 default; `voice-clone` recommends 200 | see the epochs table |
| effective batch | **8 = batch x grad_accum** on every profile | makes `steps` and `lr` mean the same on every GPU |
| grad clip | 1.0 | |
| AMP | bf16 on L4/A100, fp16 + `GradScaler` on T4, fp32 on CPU | T4/Turing has no native bf16 |
| seed | 1337 | |

Epoch arithmetic for 38 clips:

```
600 steps x 8 clips = 4800 clip-passes / 38 clips = ~126 epochs
200 steps x 8 clips = 1600 clip-passes / 38 clips = ~42 epochs
```

126 epochs over 19 minutes is a *lot*. That is why `voice-clone` recommends 200
steps, and why both projects treat "overfitting is the main failure mode" as a
first-class note: the LR is low enough that loss keeps falling while the model
memorises the clips.

## 2.6 The loss caveat (read this before you spend GPU hours)

`train_t3.py` calls the module's own loss:

```python
loss_text, loss_speech = model.t3.loss(t3_cond=..., text_tokens=..., speech_tokens=...)
```

and `T3.loss` scores position `p` against token `p` (unshifted). Generation,
however, samples the *next* token. The community kit
(`gokhaneraslan/chatterbox-finetuning`, the one behind
`msingiai/sauti_tts_v2`) instead does:

```python
speech_logits = out.speech_logits[:, :-1].transpose(1, 2)
speech_labels = speech_tokens[:, 1:]
# additionally mask the reference-prompt region and the padding out of the labels
loss_speech = F.cross_entropy(speech_logits, speech_labels, ignore_index=-100)
```

If a run produces repetitive or runaway output after training, this is the first
thing to check. `docs/03-chatterbox-finetuning.md` §1.4 has a drop-in patched
`batch_loss` that implements the shifted objective inside this repo, without
switching trainer.

## 2.7 Checkpointing, resume, and the Colab reality

**[verified]**

```
checkpoints/<run>/step_0000100/
    t3_cfg.safetensors      fp16 state_dict, same keys as the base file (~1 GB)
    training_state.json     step, losses, argv, timestamps, base file name
    run_config.json         resolved arguments
checkpoints/<run>/latest/   copy of the newest step; what --resume auto loads
```

* fp16 checkpoints (`_to_fp16_state`) keep a run inside a free Drive quota;
* **no optimiser state is saved** — resume restarts AdamW and the LR schedule from
  the saved step. Deliberate trade: ~1 GB per save instead of ~3 GB;
* `keep_last=3` rotates old step dirs (`_rotate`);
* SIGINT/SIGTERM sets a flag and the loop checkpoints *after* the current step, so
  a `--max-hours` guard (which raises SIGINT just before the session limit) costs
  at most one step;
* 50 consecutive empty batches raises `SystemExit` instead of silently looping.

## 2.8 Evaluation and export

`eval` generates the same 5 Sheng prompts from the base model and the checkpoint
(`--ab`), records duration, wall time and RTF, and writes `eval.json` + `eval.md`.
It uses a held-out clip as the voice prompt (`reference_clip` prefers the val
manifest). The judgement is human: intelligible *new* Sheng text and a
recognisable voice; "sounds right only on training sentences" = overfitting, so
take an earlier checkpoint.

`export` copies `ve.pt`, `s3gen.pt`, `grapheme_mtl_merged_expanded_v1.json`,
`conds.pt` and drops the fine-tuned `t3_cfg.safetensors` beside them, plus a
README, a load snippet and the eval WAVs. The base
`t3_mtl23ls_v2.safetensors` is deliberately **not** copied.

## 2.9 Failure modes and their signatures

| symptom | likely cause | action |
|---|---|---|
| loss falls, audio recites training text | overfit (prompt == target prefix) | fewer steps, earlier checkpoint |
| loss falls, audio is babble / never stops | LR too high, or the unshifted-loss issue | `--lr 2e-5`, §2.6 |
| loss flat at ~log(6561) = 8.79 | labels/tokens broken (all padding) | check the featuriser cache |
| val loss rises while train falls | overfit, expected | keep `valid_interval=100` |
| OOM on T4 | bf16 profile on Turing | `--profile t4` (fp16), batch*accum = 8 |
| audio quality drops (noise, clicks) | something unfroze s3gen | must never happen: assert `requires_grad=False` |

---

# Part III — Reproducing and extending

## 3.1 Minimal commands (Colab)

```bash
pip install -e ./finetune-chatterbox
sheng-chatterbox env && sheng-chatterbox data && sheng-chatterbox model
sheng-chatterbox train --profile l4 --steps 600 --max-hours 8
sheng-chatterbox eval --ab && sheng-chatterbox export
```

## 3.2 Changing the target language

1. Confirm the language tag exists in the tokenizer (2454-id vocab, 23 tags).
   Swahili is `[sw]` = 730. If the language is missing you need a vocabulary
   extension plus an embedding resize — see `docs/03` §3.3.
2. Set `language_id` in the config; it propagates to the featuriser and to
   `generate(language_id=...)`.
3. Keep the multilingual variant: the English checkpoint has a 704-id vocab and
   no `[xx]` tags at all.

## 3.3 What to measure if you must know an unknown

| unknown | experiment |
|---|---|
| optimal steps | save every 100, score each with CER + speaker cosine (`voice-clone eval --ckpt`) |
| text-dropout need | 100 steps with/without `text_emb` dropout, compare CFG on/off |
| shifted vs unshifted loss | 100 steps each, compare the `runaway` rate at `cfg_weight=0.5` |
| reference-prompt masking | same, but listen for recitation of the prompt text |

Each is a ~20 minute L4 run; none needs the full 600-step budget to show a
direction.
