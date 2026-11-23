# Fine-Tuning Chatterbox — Full, LoRA, and Component-Wise Recipes

Three ways to move Chatterbox toward a new voice or language, from most
destructive to most surgical:

| mode | trainable | data needed | risk | where it is implemented |
|---|---|---|---|---|
| **A. full T3** | 541 M (all of T3) | 10 min - 10 h | overfit / drift | `finetune-chatterbox` (existing) |
| **B. LoRA on T3** | ~27 M (5 %) | 10 min - 10 h | under-fit, adapter merge bugs | not yet implemented here; recipe below |
| **C. component-wise** | text side *or* audio side only | hours - 100 h | forgetting, broken alignment | recipe below |

All three share the property that `ve` and `s3gen` are frozen unless you
explicitly open C-part-2. The freeze matrix is the whole design:

```
                    text side  |  shared trunk  |  audio side
                    ---------- | -------------- | -----------
tokenizer (JSON)      2454 ids            --              --
text_emb / text_head    x                 --              --
T3 text_pos_emb         x                 --              --
Llama 30 layers         --              shared            --
speech_emb/head         --                --              x
speech_pos_emb          --                --              x
cond_enc (spkr/prompt/
  emotion)              --                --              x
ve                      --                --        frozen (A,B,C)
s3gen (tokenizer, CFM,
  HiFT)                 --                --       frozen unless C2
```

The trunk is genuinely shared: there is no way to fine-tune "language" without
touching parameters that also model audio. That is why the component recipes
below separate the problem along *two* axes — which parameters get gradients,
**and** which loss term is back-propagated.

---

# 1. Mode A — full T3 fine-tune (what this repo does today)

## 1.1 Shape of the run

```bash
sheng-chatterbox train --profile l4 --steps 600 --lr 5e-5 --max-hours 8
# 38 clips x 30 s, effective batch 8, AdamW, warmup 50 + cosine, bf16
```

`load_model` is the entire specification:

```python
for module_name in ("ve", "s3gen"):
    module = getattr(model, module_name); module.eval()
    for p in module.parameters(): p.requires_grad_(False)
model.t3.train()
for p in model.t3.parameters(): p.requires_grad_(True)
```

## 1.2 Memory and speed budget

| item | fp16/bf16 bytes | L4 24 GB | T4 16 GB |
|---|---|---|---|
| T3 weights | 1.08 GB | ok | ok |
| T3 gradients | 1.08 GB | ok | ok |
| AdamW m, v (fp32) | 4.33 GB | ok | ok |
| s3gen + ve (frozen) | 0.4 GB | ok | ok |
| activations (batch 4 x ~600 tok, bf16) | ~3-5 GB | ok | tight |
| **total** | | **~10-12 GB** | **~9-11 GB** (batch 2 + accum 4) |

The trainer does **not** enable gradient checkpointing (the community kit does,
for ~60 % activation savings). If you need to raise the batch on a 16 GB card,
that is the first change to make.

Throughput on an L4 is roughly 5-8 steps/s at effective batch 8 for 30 s clips
(the featuriser is cached, so this is pure T3 forward/backward), i.e. 600 steps in
~1.5-3 h.

## 1.3 What "worked" looks like

* `loss_text` falls fast (it is an easier, highly predictable target) and then
  plateaus around 0.1-0.4;
* `loss_speech` falls from ~8.8 (uniform over 6561 codes) toward 1.5-3.0;
* held-out CER drops and speaker cosine rises versus the baseline (that is what
  `voice-clone compare` measures);
* listening: new Sheng sentences are intelligible *and* the timbre is the target.

## 1.4 The loss shift — fix before trusting a long run

`T3.loss` scores position `p` against token `p`; generation samples token `p+1`.
For fine-tuning, use the shifted, prompt-masked variant. Drop this into
`train_t3.py` and call it from `batch_loss`:

```python
def shifted_losses(model, cond, text_tokens, text_lens, speech_tokens, speech_lens,
                   prompt_tokens=None):
    import torch, torch.nn.functional as F
    out = model.t3.forward(t3_cond=cond, text_tokens=text_tokens,
                           text_token_lens=text_lens, speech_tokens=speech_tokens,
                           speech_token_lens=speech_lens, training=True)

    IGNORE = -100
    dev = out.speech_logits.device

    speech_logits = out.speech_logits[:, :-1].transpose(1, 2)
    speech_labels = speech_tokens[:, 1:]
    n = speech_labels.size(1)
    pad  = torch.arange(n, device=dev)[None, :] >= (speech_lens[:, None] - 1)
    # optional: do not train on copying the reference prompt
    keep = 0 if prompt_tokens is None else prompt_tokens.size(1)
    prm  = torch.arange(n, device=dev)[None, :] < keep
    speech_labels = speech_labels.masked_fill(pad | prm, IGNORE)
    loss_speech = F.cross_entropy(speech_logits, speech_labels, ignore_index=IGNORE)

    text_logits = out.text_logits[:, :-1].transpose(1, 2)
    text_labels = text_tokens[:, 1:]
    m = text_labels.size(1)
    pad_t = torch.arange(m, device=dev)[None, :] >= (text_lens[:, None] - 1)
    text_labels = text_labels.masked_fill(pad_t, IGNORE)
    loss_text = F.cross_entropy(text_logits, text_labels, ignore_index=IGNORE)
    return loss_text + loss_speech, loss_text, loss_speech
```

Note `prompt_tokens` here means the *cond prompt* is **not** part of
`speech_tokens` in this repo's featuriser, so the mask above is a no-op for
`sheng_t3` — it matters when you build a dataset where the prompt is the prefix of
the target (the common case in other kits).

## 1.5 When full fine-tune is the wrong tool

Full T3 fine-tune changes *everything* at once: pronunciation, prosody, timbre,
pause statistics, and the mapping from the text prefix to the speech start. With
20 minutes of data it will memorise. Prefer it when:

* you have **≥ 5 h** of one speaker and want maximum fidelity;
* or you want the model to speak a language/accent that is not already in the 23
  supported ones (a new phonology needs the text side to move, §3.3).

Prefer LoRA (mode B) for anything under a few hours, which is exactly what the
community guidance says ("LoRA for anything under ~10 h").

---

# 2. Mode B — LoRA on T3

## 2.1 Why it is the right default for small data

LoRA adds `B·A` (rank `r`) to each targeted linear layer, freezes the original
weight, and therefore:

* updates a **low-rank** subspace — expressive enough for timbre/accent,
  restrained enough that it cannot rewire the language model;
* produces a **5 MB adapter** instead of a 1 GB checkpoint (a Colab Drive can hold
  dozens);
* lets you keep several voices and swap/merge them at export time.

## 2.2 Parameter arithmetic (r = 32, alpha = 64)

Per Llama layer, for `in = out = 1024` projections
(`q_proj`, `k_proj`, `v_proj`, `o_proj`):

```
A (32 x 1024) + B (1024 x 32) = 65 536
```

For the SwiGLU MLP (`gate/up`: 1024 -> 4096, `down`: 4096 -> 1024):

```
A (32 x 1024) + B (4096 x 32) = 163 840   (each of the three)
```

Per layer: `4 x 65 536 + 3 x 163 840 = 753 664`. Times 30 layers = **22.6 M**,
plus `spkr_enc` (`256 -> 1024`: 40 960) ≈ **22.65 M = 4.2 % of T3**.

If you also fully train the two embedding/head tensors
(`modules_to_save = ["text_emb", "speech_head"]`, 2.51 M each), total trainable is
**~27.7 M ≈ 5.1 %**. AdamW state for that is ~0.4 GB: a 16 GB T4 becomes viable at
batch 8.

Scaling: PEFT computes `delta = (alpha / r) * B·A`, so `r=32, alpha=64` = scale 2.
Raising `alpha` is the cheap way to make a small-rank adapter learn faster; raising
`r` is what buys capacity.

## 2.3 Wiring LoRA into this trainer

The attention/MLP module names are exactly the HF Llama ones because `T3.tfmr` *is*
`LlamaModel`; `spkr_enc` is the linear that projects the 256-d voice vector into
the LM — including it is what lets a small adapter change *who* the model sounds
like with very few parameters.

```python
# sheng_chatterbox/lora.py
from peft import LoraConfig, get_peft_model

T3_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj", "spkr_enc"]

def wrap_lora(model, r: int = 32, alpha: int = 64, dropout: float = 0.05):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=T3_LORA_TARGETS,
        modules_to_save=["text_emb", "speech_head"],   # trained fully, saved in the adapter
    )
    model.t3 = get_peft_model(model.t3, cfg)
    model.t3.print_trainable_parameters()
    return model
```

and in `train_t3.py::load_model`, after the freeze block:

```python
if args.lora:
    from .lora import wrap_lora
    model = wrap_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    trainable = [p for p in model.t3.parameters() if p.requires_grad]
else:
    trainable = list(model.t3.parameters())
optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)
```

Plus the CLI surface the sibling project already forwards:

```python
p.add_argument("--lora", action="store_true")
p.add_argument("--lora-r", type=int, default=32)
p.add_argument("--lora-alpha", type=int, default=64)
```

> **Current state:** `voice-clone`'s `train` command accepts `--lora` and forwards
> it to `sheng_chatterbox.cli train`, which does not define it — argparse exits
> with "unrecognized arguments: --lora". Adding the four lines above (and the
> checkpoint handling in §2.5) is the whole fix.

## 2.4 Hyperparameters that differ from full fine-tune

| item | full | LoRA | reason |
|---|---|---|---|
| LR | 5e-5 | **1e-4** | adapters are randomly initialised; a frozen backbone tolerates a larger step |
| steps / epochs | 600 (~126 ep) | 300-600, or `epochs x batches/8` | the adapter needs more passes to fit, but overfits later |
| batch | 8 | 8 | unchanged |
| weight decay | 0.01 | 0.0-0.01 | only on LoRA B matrices; small effect |
| warmup | 50 | 20-50 | |
| grad clip | 1.0 | 1.0 | |
| dtype | bf16 | bf16 | |

The community kit's defaults (`r=32, alpha=64, dropout=0.05, lr=1e-4, 10 epochs`)
are a sane starting point; 10 epochs on 38 clips is only ~48 steps, which is
*authored for their own multi-hour datasets*, not for 19 minutes. For Sheng,
budget 200-400 steps.

## 2.5 Checkpointing an adapter, and merging for export

`sheng_chatterbox` writes `t3_cfg.safetensors` with the full `state_dict`, which is
what `ChatterboxMultilingualTTS.from_local` loads with `strict=True`. A LoRA
adapter has entirely different key names, so it cannot be dropped in. Two
supported patterns:

**(a) save the adapter, merge on export** — recommended:

```python
# in save_checkpoint(), LoRA branch
model.t3.save_pretrained(step_dir / "adapter")          # ~5 MB, keys: ...lora_A/B
```

```python
# in infer.py::export(), LoRA branch
from peft import PeftModel
from safetensors.torch import save_file

base = ChatterboxMultilingualTTS.from_local(paths.pretrained_dir, "cpu",
                                             t3_model=MTL_T3_MODEL)
peft_t3 = PeftModel.from_pretrained(base.t3, str(ckpt / "adapter"))
merged  = peft_t3.merge_and_unload()                    # -> plain T3
save_file({k: v.half() for k, v in merged.state_dict().items()},
          str(target / "t3_cfg.safetensors"))
```

`merge_and_unload` folds `(alpha/r)·B·A` into each frozen weight and restores the
`modules_to_save` copies, so the result is a normal T3 checkpoint that needs no
PEFT at inference. This is the only way to keep the `export` contract intact.

**(b) merge every save** — simpler for the rest of the pipeline (every checkpoint
is a full T3), but each save writes ~1 GB and re-merging 3 checkpoints per run
costs a few seconds of CPU:

```python
merged = model.t3.merge_and_unload()
save_file(_to_fp16_state(merged.state_dict()), str(step_dir / CKPT_NAME))
```

Note that (b) mutates `model.t3` (the PeftModel is consumed), so keep a reference
to the adapter module if you intend to continue training after a save.

## 2.6 Judging a LoRA run

LoRA's characteristic failure is *under-fitting with confident prosody*: the voice
shifts a little, the words stay right, nothing improves. Signs and fixes:

| sign | meaning | fix |
|---|---|---|
| adapter loss barely moves below the base loss | rank too small or LR too low | `r=64`, `lr=2e-4`, include `text_emb`/`speech_head` in `modules_to_save` |
| voice changes, words degrade | over-fit / LR too high | fewer steps, `lora_dropout=0.1` |
| merged output differs from training-time output | merge bug (alpha/r, or `modules_to_save` dropped) | compare logits on one batch pre/post merge; they must match to 1e-3 |

That last check is cheap and worth scripting; silent merge bugs are the most
common way a good adapter produces bad audio.
---

# 3. Mode C — separate language-side and audio-side fine-tuning

"Language component" and "audio component" are not separate modules in this
architecture — the trunk is shared. What *is* separable is:

1. **which parameters require gradients** (text-side tables vs audio-side tables
   vs the shared trunk), and
2. **which loss term you back-propagate** (`loss_text`, `loss_speech`, or the CFM
   mel loss).

Every recipe in this section is a combination of those two levers. Nothing here
is implemented in the repo yet; the code is written to drop into
`finetune-chatterbox`.

## 3.1 The loss-level split

`T3.forward` always produces both streams, so you can choose what to learn:

```python
out = model.t3.forward(..., training=True)
loss_text   = F.cross_entropy(out.text_logits[:, :-1].transpose(1,2), text_labels,  ignore_index=-100)
loss_speech = F.cross_entropy(out.speech_logits[:, :-1].transpose(1,2), speech_labels, ignore_index=-100)

loss = {"text": loss_text, "speech": loss_speech, "both": loss_text + loss_speech}[mode]
```

* **text-only** teaches orthography -> token behaviour: how a language's strings
  should be chunked, how `[xx]` tags and non-verbal tags behave. It cannot change
  *pronunciation*, because pronunciation lives in the speech stream.
* **speech-only** teaches delivery: timbre, accent, prosody, speaking rate. It
  cannot fix a tokenizer that splits a new script badly.
* **both** (default) is what a normal SFT run does.

A practical schedule that avoids fighting yourself: 20-50% of steps text-only to
adapt the text side without disturbing audio, then switch to `both`.

## 3.2 Language-side fine-tune (case 1: language already supported)

`sw` is already id 730 in a 2454-id vocabulary, and Swahili has no special
text preprocessing (`MTLTokenizer.encode` only special-cases `zh`, `ja`, `he`,
`ko`, `ru`). So for a Swahili-family target there is nothing to fix in the
tokenizer; the language-side recipe is about *register* — Sheng slang spellings,
English content words inside Swahili grammar.

```python
# freeze everything audio-side; train the text tables + a text-only LoRA
for name in ("speech_emb", "speech_head", "speech_pos_emb", "cond_enc"):
    for p in getattr(model.t3, name).parameters():
        p.requires_grad_(False)
for name in ("text_emb", "text_head", "text_pos_emb"):
    for p in getattr(model.t3, name).parameters():
        p.requires_grad_(True)

# LoRA only on the *front* half of the trunk is an approximation: attention is
# shared, so restrict capacity instead of pretending layers are language-specific
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj", "v_proj"])   # cheapest useful subset
```

* LR: text embeddings tolerate `2e-4`, the trunk adapter `5e-5`.
* Loss: `text` for the first ~50 steps, then `both` (text weight 0.3-0.5).
* Data: transcripts matter more than audio here. 100-500 utterances of *correct*
  orthography in the target register is enough to make the tokenizer/mapping
  behaviour consistent; do not "normalise" Sheng into standard Swahili.

Effect you should expect: fewer skipped/garbled function words, better handling of
English insertions, fewer `[UNK]`. **Not** a new accent, and not new vocabulary
pronunciation.

## 3.3 Language-side fine-tune (case 2: language not in the vocabulary)

If the target language (or a non-verbal tag you need) is not in the 2454 ids, you
must extend the vocabulary. That is a bigger change than it looks, because three
things are hard-coded to 2454.

```python
# 1) tokenizer: add the new symbols and keep one id space
from tokenizers import Tokenizer
tok = Tokenizer.from_file("grapheme_mtl_merged_expanded_v1.json")
tok.add_tokens(["[lg]"] + list(new_graphemes), beginning_of_sequence=False)

# 2) resize the three text-side tensors; mean-init keeps the LM in-distribution
import torch, torch.nn as nn
old = model.t3.text_emb.num_embeddings
new_ids = tok.get_vocab_size()
model.t3.text_emb = nn.Embedding(new_ids, 1024)
model.t3.text_head = nn.Linear(1024, new_ids, bias=False)
with torch.no_grad():
    model.t3.text_emb.weight[:old] = old_state["text_emb.weight"]
    model.t3.text_emb.weight[old:] = old_state["text_emb.weight"].mean(0, keepdim=True)
    model.t3.text_head.weight[:old] = old_state["text_head.weight"]
    model.t3.text_head.weight[old:] = old_state["text_head.weight"].mean(0, keepdim=True)

# 3) THE TRAP: alignment watchdog switches itself off
model.t3.hp.text_tokens_dict_size = new_ids
#   T3Config.is_multilingual == (text_tokens_dict_size == 2454)
#   -> False, so AlignmentStreamAnalyzer is never attached and long texts can
#      run away. Patch it explicitly:
model.t3.hp._is_multilingual = True
T3Config.is_multilingual = property(lambda self: True)
```

Also in the trainer: `hp.start_text_token`/`stop_text_token` are 255/0 and the
tokenizer's `[START]`/`[STOP]` are the same ids, so extending at the *end* of the
id space is safe; inserting tokens in the middle silently corrupts every prompt.

Recipe: freeze everything, train only the new rows for ~100 steps with
`loss_text`, then unfreeze `text_emb`/`text_head` + LoRA `q/k/v/o` and train
`both` at `5e-5`. Data: ≥ 5 h for a genuinely new phonology is the realistic bar;
anything less and the model will read the new script with the nearest existing
phonology, which is usually worse than transliterating into Latin script and
staying in-distribution.

## 3.4 Audio-side fine-tune (identity / timbre, no text changes)

The audio side of T3 is `speech_emb`, `speech_head`, `speech_pos_emb` and
`cond_enc`. `cond_enc` is where identity enters:

```
speaker_emb (256) --spkr_enc--> (1, 1024)     \
prompt_tokens (150) --perceiver--> (32, 1024)  >-- prefix --trunk
emotion_adv (scalar) --fc--> (1, 1024)        /
```

So "audio component fine-tuning" means one of two things:

**(a) conditioning-only (cheap, safe, limited).** Train `cond_enc.spkr_enc` and
`cond_enc.perceiver` plus `speech_head`, with the trunk frozen and
`loss = loss_speech`:

```python
trainable = ["cond_enc", "speech_head", "speech_emb"]
```

This can re-map *how* a 256-d voice vector is interpreted, and re-weight which
latents the perceiver keeps. It cannot change prosody much — that lives in the
trunk. Useful when you have seconds-to-minutes of a new speaker and want to nudge
identity without any risk to language behaviour. LR `1e-3` is safe here because
so little is trainable; 200-400 steps.

**(b) trunk LoRA restricted to the audio path.** LoRA on `gate/up/down_proj`
(MLP, where acoustic detail tends to live) plus full `spkr_enc` + `speech_head`,
trained with `loss = loss_speech` and text embeddings frozen:

```python
LoraConfig(r=32, lora_alpha=64, target_modules=["gate_proj","up_proj","down_proj","spkr_enc"],
           modules_to_save=["speech_head"], bias="none")
```

This is the honest "tune the audio component" for a voice-conversion goal:
the text side is untouched, so pronunciation and vocabulary are preserved, while
timbre/prosody move. Risk: without the text loss, alignment slowly degrades after
many steps — keep it short (≤ 300) and monitor CER on the fixed prompt set.

## 3.5 s3gen fine-tuning (the actual *audio model*)

Only worth doing if you have **≥ 1-5 h of 24 kHz** audio from the target domain.
Two bad outcomes are guaranteed otherwise: the CFM collapses to a blur, or the
HiFT vocoder learns the recording channel (clicks, noise) as part of the voice.
The repo's 16 kHz training WAVs are **not** usable here.

What to unfreeze, in order of risk:

| piece | unfreeze? | why |
|---|---|---|
| `flow.encoder` (UpsampleConformer) | no | tokens -> content; a small dataset will corrupt alignment |
| `flow.encoder_proj` | optional | cheap adapter on the mel-rate features |
| `flow.spk_embed_affine_layer` | yes | 192 -> 80 identity projection, 15 k params |
| `flow.decoder` (velocity estimator) | yes | where the timbre/texture is actually generated |
| `mel2wav` (HiFT) | **last**, or never | needs adversarial training; at this data scale it only adds artefacts |

Data prep must reproduce `embed_ref` exactly: 24 kHz mono, `n_fft 1920`,
`hop 480`, 80 mels, `fmax 8000`, 2 mel frames per 40 ms token; prompt window up to
10 s (`DEC_COND_LEN`), `prompt_feat` from the reference, `prompt_token` from the
16 kHz copy.

The loss is already in the package (`flow_matching.compute_loss`), so a minimal
trainer is:

```python
for batch in loader:                        # x1 = target mel (1, 80, T)
    mu    = encoder_proj(encoder(input_embedding(cat([prompt_token, tokens]))))
    spks  = spk_embed_affine_layer(F.normalize(campplus(wav16k), dim=1))
    cond  = cat([prompt_feat, zeros(...)])[:, :, :T]
    loss  = cfm.compute_loss(x1, mask, mu=mu, spks=spks, cond=cond)   # uses training_cfg_rate=0.2
```

* LR `1e-4` for `decoder` + affine layer, `1e-5` if you also open `encoder_proj`;
* batch 4-8 clips of 5-15 s, gradient clip 1.0, 1 000-5 000 steps;
* monitor (i) mel L1 on held-out clips, (ii) speaker cosine of the *output*-vs-
  reference through ECAPA (not through `ve` — that is a different embedding space),
  (iii) a spectral-distance sanity check;
* always regenerate through the frozen HiFT and listen for the channel-noise
  tell-tale (a fixed hiss or room tone that follows the reference, not the text).

The frozen-HiFT assumption is also your safety net: whatever the CFM does, the
vocoder's F0/harmonic structure is unchanged, so degraded output usually means
"bad mel", not "bad voice".

## 3.6 Choosing a mode

| goal | data | mode | loss | trainable |
|---|---|---|---|---|
| clone one speaker (this repo) | 20 min - 5 h | A (full T3) or B (LoRA) | both | all T3 / adapters |
| 2-10 voices, keep them separate | 10-60 min each | B | both | 5 MB adapter per voice |
| fix orthography / new tags, keep voice | 100-500 utts text | C 3.2 | text then both | text tables |
| new script / new phonology | 5 h+ | C 3.3 | text warmup then both | text tables + LoRA |
| nudge identity, tiny data | seconds-minutes | C 3.4a | speech | `cond_enc`, `speech_head` |
| preserve pronunciation, change timbre | 1 h+ | C 3.4b | speech | MLP LoRA + `spkr_enc` |
| repair/adapt the vocoder to a domain | 1-5 h @ 24 kHz | C 3.5 | CFM MSE + (HiFT adv.) | `flow.decoder` |

Rule of thumb that falls out of the parameter counts: **if your data would not
fill a single epoch of a 541 M-parameter model, do not give it that many
parameters.** 20 minutes is 4800 clip-passes at effective batch 8 — enough to
memorise, not enough to generalise, so constrain the update (LoRA) or the
supervised surface (component).

---

# 4. Guard rails that apply to all modes

## 4.1 Fix the evaluation before you train

You cannot compare two runs whose prompts changed. `voice-clone` already enforces
this: 7 frozen prompts, 3 buckets (`sheng`, `swahili`, `english`), one reference
clip from validation only. Keep that invariant for every mode here.

Minimum report per checkpoint:

```
ok clips (not silent / not clipped / not runaway)   baseline vs tuned
CER mean (ASR judge, whisper-small)                 baseline vs tuned
speaker cosine (ECAPA vs reference clip)            baseline vs tuned
UTMOS or a MOS proxy, if you have it
```

Verdict logic already implemented in `metrics.judge`: `KEEP` only when the words
did not get worse *and* the speaker got closer; `REJECT` when either flips badly;
`MIXED` otherwise — and `MIXED` is the normal outcome for a 20-minute fine-tune.

## 4.2 The three-checkpoint rule

Save at 100-step intervals, listen to three of them (early / middle / last), and
pick the *earliest* that sounds like the target. Every symptom of overfitting is
audible before it is measurable:

* reciting training sentences for any prompt;
* a rising `[STOP]` rate (clips end mid-word) or a falling one (runaway);
* timbre that is *more* like the reference clip than the dataset average — that is
  prompt leakage, not cloning.

## 4.3 Determinism

`generate` samples (`do_sample=True`, temperature 0.8, min_p 0.05). Two eval runs
with different seeds are not comparable clip-by-clip; `voice-clone` seeds with
1337 for exactly this reason. Set `torch.manual_seed(seed)` before synthesis, and
prefer comparing aggregate CER/cosine over 7 prompts rather than a single clip.

## 4.4 What not to do

* Do not fine-tune `ve`. It is the *measurement* of identity in the other project's
  scorer; training it turns evaluation into self-fulfilment.
* Do not change `exaggeration` between training and inference (0.5 both).
* Do not train on 24 kHz audio with the 16 kHz featuriser, or vice versa — the S3
  tokenizer's rate assumptions are strict and the failure is silent.
* Do not ship a watermarked-vs-unwatermarked comparison as a metric delta.

---

# 5. Appendix — checklists

## 5.1 Adding LoRA to `finetune-chatterbox` (complete)

```text
[ ] new file src/sheng_chatterbox/lora.py         (config + wrap_lora)
[ ] train_t3.build_parser: --lora, --lora-r, --lora-alpha
[ ] train_t3.load_model: wrap after freezing ve/s3gen
[ ] train_t3.train: optimizer over requires_grad params only (already does)
[ ] train_t3.save_checkpoint: LoRA -> save_pretrained(step/adapter)
[ ] infer.load_finetuned: detect step/adapter, PeftModel.from_pretrained, merge
[ ] infer.export: merge before writing t3_cfg.safetensors
[ ] voice-clone/train: stop forwarding --lora (or keep, now that it exists)
[ ] test: merged logits == training logits on one fixed batch (rtol 1e-3)
```

## 5.2 Verifying a merge

```python
import torch
with torch.no_grad():
    a = peft_model.forward(**batch)          # adapter active
    b = merged.forward(**batch)              # folded weights
for k in ("speech_logits", "text_logits"):
    assert torch.allclose(a[k].float(), b[k].float(), atol=1e-3), k
```

If this fails, the fold used the wrong `alpha/r`, or `modules_to_save` tensors were
dropped during the merge. Bad audio from a "good" adapter is almost always this.

## 5.3 Reference hyperparameters, all modes

| | full | LoRA | text-side | audio-side | s3gen |
|---|---|---|---|---|---|
| LR | 5e-5 | 1e-4 | 2e-4 (emb) / 5e-5 (adapter) | 1e-3 (cond only) / 1e-4 | 1e-4 |
| steps | 200-600 | 200-600 | 100-300 | 200-400 | 1k-5k |
| effective batch | 8 | 8 | 8 | 8 | 4-8 clips |
| warmup | 50 | 20-50 | 10 | 20 | 100 |
| clip | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| dtype | bf16 | bf16 | bf16 | bf16 | fp32/bf16 |
| checkpoint | 1 GB | 5 MB | 5-40 MB | 30 MB | 700 MB |
