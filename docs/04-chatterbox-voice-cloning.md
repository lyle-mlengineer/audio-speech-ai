# How Voice Cloning Works in Chatterbox

Chatterbox clones a voice **zero-shot, with no gradient steps**: you give it 6-10
seconds of reference audio and it speaks your text in that voice. This document
explains the mechanism precisely — which tensors are extracted from the reference,
where each one enters the network, which knobs actually control identity, and how
to tell a good clone from a broken one.

It is the "how it works" companion to `voice-clone/README.md` (the operational
runbook) and to `01-chatterbox-multilingual-architecture.md` (the full tensor map).

---

## 1. The one-sentence version

> Three independent conditionings are extracted from the reference clip — a
> 256-d voice vector for the LM, 150 speech tokens compressed to 32 latents for
> in-context prosody, and a mel + 192-d x-vector for the flow-matching decoder —
> and generation is then conditioned on all three during sampling. Nothing is
> trained, so the reference *is* the model state for that utterance.

---

## 2. The entry point

```python
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
model.prepare_conditionals("ref.wav", exaggeration=0.5)      # 1. clone
wav = model.generate("Sasa wewe, unajua vile mambo iko.",     # 2. speak
                     language_id="sw", exaggeration=0.5,
                     cfg_weight=0.5, temperature=0.8,
                     repetition_penalty=2.0, min_p=0.05, top_p=1.0)
```

`generate(..., audio_prompt_path="ref.wav")` is the same thing with the two calls
fused. `prepare_conditionals` is where all cloning happens — it fills
`model.conds`, a `Conditionals(t3: T3Cond, gen: dict)` object, and every later
call reuses it. `conds.pt` in the released repo is simply *a saved instance of
this object* for Resemble's built-in voice, which is why the model can speak
without you supplying a reference.

Internally:

```
prepare_conditionals(wav_fpath, exaggeration=0.5)
├── librosa.load(wav, sr=24000)                        -> s3gen_ref_wav
├── librosa.resample(-> 16000)                         -> ref_16k_wav
├── s3gen_ref_wav = s3gen_ref_wav[:10 s]               DEC_COND_LEN
├── s3gen.embed_ref(s3gen_ref_wav, 24000)              -> gen dict (§3.2-3.4)
├── s3gen.tokenizer(ref_16k_wav[:6 s], max_len=150)    -> t3_cond_prompt_tokens
├── ve.embeds_from_wavs([ref_16k_wav])                 -> t3_cond.speaker_emb
└── model.conds = Conditionals(T3Cond(...), ref_dict)
```

Everything is deterministic and cheap (no LM forward), which is why the reference
can be swapped per request in a server without reloading 2 GB of weights.

---

## 3. The five tensors extracted from the reference

| # | tensor | extracted from | shape | consumed by | role |
|---|---|---|---|---|---|
| 1 | `speaker_emb` | 16 kHz, `ve` (GE2E) | `(1, 256)` | `T3CondEnc.spkr_enc` -> `(1, 1024)` prefix position 0 | "who" at the semantic level |
| 2 | `cond_prompt_speech_tokens` | 16 kHz first 6 s -> S3 tokenizer | `(1, 150)` | `speech_emb` + `Perceiver` -> `(1, 32, 1024)` prefix positions 2..33 | "how this speaker talks" in-context |
| 3 | `emotion_adv` | the `exaggeration` argument | `(1, 1, 1)` | `emotion_adv_fc` -> `(1, 1024)` prefix position 34 | intensity knob |
| 4 | `prompt_feat` | 24 kHz first 10 s -> 80-band mel @ 50 fps | `(1, 80, 2*150)` | CFM `cond` (prepended frames) | acoustic texture the decoder continues |
| 5 | `embedding` | 16 kHz, CAMPPlus x-vector | `(1, 192)` | `spk_embed_affine_layer` -> `(1, 80)`, CFM `spks` | "who" at the waveform level |
| — | `prompt_token` | 16 kHz -> S3 tokenizer | `(1, 150)` | `S3Token2Mel.encoder` prefix | warms the conformer's receptive field |

Two facts follow immediately from this table and explain most cloning behaviour:

* **T3 never sees the reference audio, only 32 latents and a 256-d vector.** It
  cannot copy the reference's waveform, only its statistics — so a cloned clip can
  sound like the speaker without being a resynthesis of the prompt.
* **S3Gen sees the reference *mel* directly.** That is what makes the output
  inherit channel characteristics (mic, room, codec) as well as voice. A noisy
  reference gives a noisy clone; a reference with music gives a clone with music.

### 3.1 Why two different speaker embeddings

| embedding | dim | rate | consumer | property |
|---|---|---|---|---|
| `ve` (Resemblyzer/GE2E) | 256 | 16 kHz | T3 conditioner | trained for speaker discrimination, robust to content |
| CAMPPlus x-vector | 192 | 16 kHz | s3gen flow | trained jointly with the vocoder, captures timbre+channel |

They are not interchangeable and not comparable numerically. `voice-clone`'s scorer
uses a **third** space (SpeechBrain ECAPA) precisely so that measuring the clone
does not use the same encoder the model was conditioned with.

### 3.2 Rate contract for the reference

```
prompt_feat  : 6 s @ 24 kHz -> 300 mel frames @ 50 fps   (hop 480, n_fft 1920, fmax 8 kHz)
prompt_token : 6 s @ 16 kHz -> 150 tokens @ 25 Hz        (40 ms each)
2 * 150 = 300 mel frames                                 <- the contract
```

`embed_ref` enforces `mel_len == 2 * token_len` and truncates the tokens if the
reference was not padded to a 40 ms multiple. If your reference is, say, 6.02 s,
you get a one-line warning and a slightly truncated prompt — harmless, but it is
why `voice-clone`'s resampler writes exact-length WAVs.

### 3.3 The 6 s / 10 s split

```python
ENC_COND_LEN = 6 * S3_SR      # 96000 samples  -> T3 prompt tokens
DEC_COND_LEN = 10 * S3GEN_SR  # 240000 samples -> s3gen mel + x-vector
```

The two halves of the model want different amounts of context, and the caps are
independent:

* the LM only pays a fixed **34 prefix positions** no matter how long the prompt
  is (32 latents from the perceiver), so a longer T3 prompt mostly adds resolution
  to those latents, not sequence length;
* s3gen's cost grows with `prompt_feat` frames (they are part of the CFM's
  conditioning length), so >10 s is simply dropped.

Practical consequence: **6 s is the useful minimum, 10 s the useful maximum**, and
everything past 10 s is silently ignored. `prepare_conditionals` emits
`WARNING: s3gen received ref longer than 10s` for a reason.

### 3.4 Non-obvious detail: `ve` averages windows

```python
ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
ve_embed = ve_embed.mean(axis=0, keepdim=True)     # (1, 256)
```

`embeds_from_wavs` produces one embedding per ~1.6 s window; `prepare_conditionals`
averages them. A reference that contains two speakers therefore yields the
*average* voice — a classic silent failure (the clone sounds like neither). Use one
speaker, no overlapping speech, no music.

---

## 4. Where the conditionings enter the network

```
                          ref.wav
              ┌───────────────┼────────────────────┐
        16 kHz│         16 kHz│              24 kHz│
              ▼               ▼                    ▼
        ve -> (256)   S3 tokzr -> (150)      mel -> (80, 300)
              │               │                    │
       spkr_enc(1024)   perceiver(32x1024)    prompt_feat
              │               │                    │
              └──── T3 prefix (34 positions) ──────┤
                              │                    │
                        text tokens                │
                              ▼                    │
                      [ T3 samples speech tokens ] │
                              │                    │
                              ▼                    ▼
                      S3 tokens ──► conformer ──► CFM <── x-vector (192) -> (80)
                                                  │
                                             HiFT -> 24 kHz
```

Read it as three nested levels of "sameness":

1. **T3, prefix level**: the LM is told "this is the speaker and how they speak"
   before it sees any text. Nothing in the prompt is copied into the output —
   the prompt is *context*, and the LM's job is to continue in the same style.
2. **CFM, conditioning level**: the decoder is given the reference mel as the
   past it is expected to continue (`conds[:, :mel_len1] = prompt_feat`), and the
   x-vector as a global timbre bias.
3. **Guidance, amplification level**: both halves use classifier-free guidance to
   push toward the conditioned answer (text guidance in T3, speaker guidance in
   the CFM).

### 4.1 Why in-context conditioning works for speech

The 150 prompt tokens are embedded into the *same* space as the tokens T3 is about
to emit, and placed immediately before the text block. To the attention stack this
looks like "a recording in this voice, then a transcript, then more of the same
recording". Because the LM was trained with exactly this layout (prompt -> text ->
continuation), voice cloning is a *sequence continuation* problem rather than a
speaker-adaptation problem. That is why 6 seconds suffice and why the model needs
no speaker embeddings at all — the prompt *is* the embedding, compressed 150 -> 32
by the perceiver so that it stays cheap.

### 4.2 The two guidances are not the same knob

| guidance | where | applies to | default | effect of raising |
|---|---|---|---|---|
| `cfg_weight` | `T3.inference` | text conditioning (row 1 has `text_emb` zeroed) | 0.5 | text adherence up, prosody flatter, more brittle |
| `inference_cfg_rate` | `flow_matching.solve_euler` | speaker (row 2 has `spks` zeroed) | 0.7 (fixed in code) | stronger reference identity, more reference *leakage*, brittle prosody |

Only `cfg_weight` is exposed by the public API. The 0.7 in the vocoder is a
constant; if you want to change it you must patch `CFM_PARAMS`.

---

## 5. Doing it well: reference clip rules

The clone is a function of the reference, so the reference **is** the product
specification. Rules that follow from §3:

| rule | why | symptom if broken |
|---|---|---|
| 6-10 s, single speaker, no overlap | `ve` averages windows; `prompt_feat` is 10 s max | clone sounds like an average of people |
| the *same* language/accent as the target text | prompt tokens carry phonology into the LM | Swahili prompt + English text = accent bleed |
| clean, dry, no music | `prompt_feat` is used as CFM conditioning | music or room tone in the output |
| consistent recording channel with the desired output | 24 kHz mel carries channel | clone sounds like a phone call when you wanted studio |
| no clipping, no long silences | S3 tokenizer emits silence tokens | fade-in eats the first word, odd pacing |
| `exaggeration` consistent between conditioning and generation | it is a conditioning input, not a sampler knob | voice shifts mid-session |

The repository encodes two of these in code:

* `voice-clone/dataset.py::pick_reference` picks the **longest validation clip** —
  longest (most speaker in one file) and never trained on (no leakage into the
  fine-tune);
* `data` stage writes the reference's transcript to `ref_text.txt` next to
  `ref_audio.wav`, so the eval harness can compare against what the speaker
  actually said.

### 5.1 Reference leakage, and why it is the interesting failure

Because the CFM is conditioned on the reference mel *as preceding audio*, an
over-strong speaker guidance or a very short reference makes the output start as a
continuation of the prompt: you hear the prompt's last syllable, or the same
intonation contour every time, or the pitch of the prompt's first word. Two
mechanical guards exist:

* `S3Gen` applies a **20 ms cosine fade-in** to every output
  (`output_wavs[:, :len(trim_fade)] *= trim_fade`) explicitly to suppress
  "spillover" from the reference clip;
* `AlignmentStreamAnalyzer.false_start` delays committing the first speech frames
  when attention activations look off-diagonal (i.e. the model is "reading" the
  prompt instead of the text).

If you still hear leakage: shorten the reference to 6 s, lower `cfg_weight`, or
raise `n_cfm_timesteps` (10 -> 16) to give the CFM more integration steps.

---

## 6. Sampling knobs, with the actual defaults

`ChatterboxMultilingualTTS.generate` -> `T3.inference`:

```python
temperature         = 0.8      # scale before warpers
cfg_weight          = 0.5      # text CFG (0 = ignore text guidance branch)
repetition_penalty  = 2.0      # note: high; 1.1-1.2 is the usual range
min_p               = 0.05     # drop tokens below 5% of the top probability
top_p               = 1.0      # effectively off by default
max_new_tokens      = 1000     # hard cap ~40 s of speech
```

| knob | raise it | lower it |
|---|---|---|
| `temperature` | more variety, more word errors | flatter, more robotic, more reliable |
| `cfg_weight` | follows the text harder, more monotone | looser, more expressive, may drift from text |
| `repetition_penalty` | fewer loops, more clipped/short | loops and stutter return |
| `min_p` | safe but bland | more variety, more chance of invalid tokens |
| `n_cfm_timesteps` | cleaner mel, slower | artefacts, faster |

`repetition_penalty=2.0` is aggressive for an LM but appropriate here: T3 outputs
40 ms tokens, and without it, repeated tokens produce audible stutter.

---

## 7. Evaluating a clone (what each number can and cannot tell you)

A cloned clip fails in exactly three ways, and only one is a training problem:

| failure | measurable signal | tool | fixed by |
|---|---|---|---|
| broken audio (silence, clipping, runaway) | RMS < 1e-3, >1 % samples at full scale, duration >> expectation | waveform only, no model | inference params, checkpoint choice |
| wrong words | CER/WER vs the prompt (ASR judge) | `openai/whisper-small` in `voice-clone` | fewer steps / lower LR |
| wrong voice | cosine of ECAPA embeddings vs the reference clip | `speechbrain/spkrec-ecapa-voxceleb` | better reference, better fine-tune, more guidance |

`voice-clone/metrics.py` thresholds these: `SILENT_RMS = 1e-3`,
`CLIP_FRAC = 0.01`, `CER_BAD = 0.60` / `CER_GOOD = 0.35`,
`SIM_BAD = 0.50` / `SIM_GOOD = 0.65`. The verdict (`KEEP` / `MIXED` / `REJECT` /
`NO GAIN`) requires *both* directions to agree; `MIXED` is the honest normal case
for 20 minutes of data.

Two traps:

* **Do not mix embedding spaces.** `ve` (256-d, used by T3), CAMPPlus (192-d,
  used by s3gen) and ECAPA (192-d, used to score) are three different metrics. A
  clone can be "close" in one and not another, and a cosine of 0.7 in ECAPA means
  something different from 0.7 in `ve`.
* **Do not score against a clip that was trained on.** That produces a great number
  and a bad model. The reference clip must come from `val`/`test`.

For human evaluation, an ABX test with 10+ listeners on 7 prompts tells you more
than any single automatic metric; the automatic numbers are for *ranking your own
runs*, not for absolute quality claims.

---

## 8. Zero-shot vs fine-tuned cloning

| | zero-shot (`prepare_conditionals` only) | fine-tuned T3 (this repo) |
|---|---|---|
| data | 6-10 s | 20 min - 5 h |
| what it captures | timbre, rough prosody, channel | timbre + accent + code-switch delivery + speaking rate |
| failure mode | generic accent, mild drift | memorisation, runaway, unstable prosody |
| setup cost | none | 1-3 h GPU |
| typical use | "sound like this file" | "sound like this speaker, in this dialect" |

They compose: a fine-tuned T3 **plus** a fresh reference clip from the same
speaker is the best of both, and it is what `voice-clone eval --ckpt` does — the
checkpoint supplies the dialect, the prompt supplies the specific recording's
timbre/channel.

The risk of composing them is **prompt-target correlation**: in training,
`cond_prompt_speech_tokens` are the first 6 s of the *same* clip being predicted.
Train long enough and the model learns "repeat what follows the prompt", which at
inference means reciting a training clip. Mitigations, in order of cost:

1. stop earlier (200 steps, not 600);
2. use a held-out reference clip for eval and for any listening test;
3. train with prompt dropout (replace the prompt with zeros 10-20 % of steps);
4. use a prompt taken from a *different* clip of the same speaker during training
   (the cleanest fix, and what most cloning recipes do).

---

## 9. Rights, watermarking, and honest limits

* Every output passes through `perth.PerthImplicitWatermarker` inside `generate`
  — cloned audio is watermarked whether or not you want it.
* The Sheng dataset (`lyle-mlengineer/kenyan-celebs`) is derived from third-party
  video; check rights before publishing an exported voice.
* Chatterbox is archived research (MIT, unmaintained since Aug 2026): weights are
  public forever, but no upstream fixes. Pin `chatterbox-tts` and keep the
  pretrained files in your artifact store.
* A 20-minute fine-tune is a **voice** adaptation, not a language course: new Sheng
  slang that the base model has never seen will still be mispronounced. Keep
  prompts within the dataset's register (Swahili orthography + English content
  words) and measure with prompts you never trained on.

---

## 10. Minimal working example, annotated

```python
import torch, torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_local(
    "pretrained/chatterbox", device="cuda", t3_model="t3_cfg.safetensors"   # or the base file
)

# --- cloning: 1 forward-free call, fills model.conds ---
model.prepare_conditionals("ref_audio.wav", exaggeration=0.5)
#   model.conds.t3.speaker_emb                 (1, 256)
#   model.conds.t3.cond_prompt_speech_tokens   (1, 150)
#   model.conds.t3.emotion_adv                 (1, 1, 1)
#   model.conds.gen["prompt_feat"]             (1, 80, 300)
#   model.conds.gen["embedding"]               (1, 192)
#   model.conds.gen["prompt_token"]            (1, 150)

torch.manual_seed(1337)                      # sampling is stochastic
with torch.inference_mode():
    wav = model.generate(
        "Sasa wewe, unajua vile mambo iko.",
        language_id="sw",                    # [sw] = id 730, no preprocessing
        exaggeration=0.5,                    # must match prepare_conditionals
        cfg_weight=0.5, temperature=0.8,
        repetition_penalty=2.0, min_p=0.05,
    )                                        # (1, N) float32 @ 24 kHz, watermarked

torchaudio.save("out.wav", wav.cpu(), model.sr)
```

To clone a *different* voice in the same process, call `prepare_conditionals`
again — there is nothing else to reset. To go back to the built-in voice, reload
`conds.pt` (or call `generate(audio_prompt_path=...)` with the reference).
