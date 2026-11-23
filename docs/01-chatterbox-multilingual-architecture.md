# Chatterbox Multilingual — Architecture and Tensor Flow Walkthrough

A line-by-line structural map of `ResembleAI/chatterbox` (the **multilingual**,
23-language checkpoint this repository fine-tunes) and a complete trace of the
tensors from a text string to a 24 kHz waveform.

Everything here is read from the shipped source, not from memory:

| source | what it is |
|---|---|
| `chatterbox-tts==0.1.7` (PyPI wheel) | the version whose weight filenames match `finetune-chatterbox` (`t3_mtl23ls_v2.safetensors`, `grapheme_mtl_merged_expanded_v1.json`) |
| `resemble-ai/chatterbox` @ `master` | current `mtl_tts.py` — adds the `t3_model=` argument that `sheng_chatterbox` uses |
| `grapheme_mtl_merged_expanded_v1.json` | the real multilingual text vocabulary (2454 entries; read directly) |

Field names in `code font` are exact identifiers, so you can grep them:

```bash
python -c "import chatterbox, os; print(os.path.dirname(chatterbox.__file__))"
```

---

## 1. The four artifacts

Chatterbox is not one network. It is four frozen artifacts plus one small
conditioning object, and the repo's fine-tune touches exactly one of them.

```
ResembleAI/chatterbox/                (~2 GB, MIT)
├── t3_mtl23ls_v2.safetensors        T3: 0.5B Llama  -> text tokens to speech tokens
├── s3gen.pt                         S3Gen: S3 tokens -> 24 kHz waveform
│     ├── tokenizer.*                S3 speech tokenizer (6561 codes @ 25 Hz)
│     ├── speaker_encoder.*          CAMPPlus x-vector (192-d) for the decoder
│     ├── flow.*                     CausalMaskedDiffWithXvec (conformer + CFM)
│     └── mel2wav.*                  HiFTGenerator (HiFi-GAN + F0 harmonic source)
├── ve.pt                            voice encoder (Resemblyzer GE2E, 256-d)
├── grapheme_mtl_merged_expanded_v1.json   BPE over characters, 2454 ids, 23 [lang] tags
└── conds.pt                         one built-in voice (Conditionals: T3Cond + s3gen ref_dict)
```

| module | params | trainable in `finetune-chatterbox` | trainable in `voice-clone` |
|---|---|---|---|
| `t3` | ~522 M | **yes (full)** | yes (via the same trainer) |
| `s3gen` (tokenizer + CFM + HiFT + x-vector) | ~200 M | frozen | frozen |
| `ve` (voice encoder) | ~4.5 M | frozen | frozen |
| tokenizer | 0 (JSON) | frozen | frozen |

`train_t3.py` freezes `ve` and `s3gen` and trains only `t3`:

```python
for module_name in ("ve", "s3gen"):
    module = getattr(model, module_name)
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
model.t3.train()
for param in model.t3.parameters():
    param.requires_grad_(True)
```

That single decision is the reason the whole pipeline below is safe to fine-tune:
no matter how badly the LM is trained, the vocoder still renders clean 24 kHz
audio — a broken T3 produces *wrong tokens*, not *noise*.

---

## 2. End-to-end block diagram

Rates are the contract that keeps the two halves glued together.

```
            "Sasa wewe, unajua..."          ref_clip.wav (6 s)
                     │                            │
        ┌────────────▼─────────────┐              │
        │ MTLTokenizer (BPE/char)  │              │
        │ + [sw] tag               │              │
        └────────────┬─────────────┘              │
                     │ ids  (1, L+2)              │
                     │                            │
   ┌─────────────────▼──────────────────┐         │
   │ T3  (Llama-520M, 30 layers)        │◄────────┤ ve: 256-d speaker embedding
   │  cond 34 + text L+2 + speech S+1   │◄────────┤ 150 S3 prompt tokens
   └─────────────────┬──────────────────┘         │
                     │ speech ids (S,) @ 25 Hz    │
                     │                            │
   ┌─────────────────▼──────────────────┐         │
   │ S3Gen.tokenizer  : wav -> 6561     │         │
   │ S3Token2Mel      : ids -> 80-d mel │◄────────┘ s3gen.embed_ref: prompt_feat (80-d mel),
   │ CFM (Euler, 10 steps)              │            192-d CAMPPlus x-vector, prompt_token
   └─────────────────┬──────────────────┘
                     │ mel (80, 50 fps)
   ┌─────────────────▼──────────────────┐
   │ HiFTGenerator: mel -> 24 kHz wav   │   (upsample 8x5x3 = 120)
   └─────────────────┬──────────────────┘
                     │
              Perth watermarker (48 kHz, inaudible)
                     │
                 torch.FloatTensor (1, N) @ 24 kHz
```

Rate table — memorise this, every shape below follows from it:

| quantity | rate | source |
|---|---|---|
| text ids | 1 id ≈ 1.35 chars (char BPE) | `grapheme_mtl_merged_expanded_v1.json` |
| S3 speech tokens | **25 /s** (40 ms each) | `S3_TOKEN_RATE = 25` |
| mel frames (s3gen) | **50 /s** (`n_fft 1920`, `hop 480` @ 24 kHz, 80 mels) | `utils/mel.py` |
| mel frames per token | **2** (`token_mel_ratio = 2`) -> 25 Hz x 2 = 50 fps | `flow.py:52` |
| waveform | **24 kHz** = 50 fps x 120 x 4 (`istft hop_len = 4`) | `S3GEN_SR`, `hifigan.py` |
| T3 input audio | **16 kHz** (`ve`, S3 tokenizer) | `S3_SR = 16000` |
| reference window | **6 s** for T3, **10 s** for s3gen | `ENC_COND_LEN`, `DEC_COND_LEN` |

---

## 3. Stage A — text to token ids

`MTLTokenizer` (`models/tokenizers/tokenizer.py`) is a HuggingFace `tokenizers`
file: a **BPE whose base symbols are individual characters**, with no
pre-tokenizer, no normalizer, no decoder configured.

`MTLTokenizer.encode` does, in order:

1. `lowercase=True`, `NFKD` normalisation (`preprocess_text`);
2. language-specific preprocessing — only for `zh` (Cangjie), `ja` (kanji ->
   hiragana), `he` (diacritics), `ko`, `ru` (stress marks). **Swahili has none**,
   so `sw` is a pure lowercase+NFKD path;
3. prepend the literal language tag, `f"[{language_id}]"` -> `"[sw]"`;
4. replace every space with the literal token `[SPACE]`;
5. BPE-encode.

Vocabulary facts read from the file itself:

| id | token | note |
|---|---|---|
| 0 | `[STOP]` | also `hp.stop_text_token` and the pad id |
| 1 | `[UNK]` | |
| 2 | `[SPACE]` | stands in for `" "` (so no token is "empty") |
| 255 | `[START]` | also `hp.start_text_token` |
| 730 | `[sw]` | Swahili tag (`[en]` = 708) |
| 2453 | `ạ` | last id -> `text_tokens_dict_size = 2454` |
| — | `[laughter]`, `[sigh]`, `[inhale]`, `[whisper]`, `[gasp]`, `[guffaw]`, … | 118 `added_tokens`: language tags **and non-verbal events**; `[ipa]` switches the tokenizer into phoneme mode |

The English-only checkpoint uses the same architecture with a 704-entry
character vocab (`T3Config.english_only()`); `T3Config.is_multilingual` is
literally `text_tokens_dict_size == 2454`, and it only switches on the
inference-time alignment hook (§5.3).

### Worked example (ids read from the shipped vocab)

Input:

```
Sasa wewe, unajua vile mambo iko. Let me tell you, hii ni serious.
```

After lowercase + NFKD + tag + space replacement, the encoder emits 51 ids:

```
730 [sw]      32 s       55 as      14 a       2 [SPACE]  100 we     100 we
  7 ,          2 [SPACE] 97 un     14 a      23 j        34 u       14 a
  2 [SPACE]   35 v       22 i       64 le      2 [SPACE] 26 m      122 am
165 bo         2 [SPACE] 22 i       24 k       28 o        9 .        2 [SPACE]
 64 le        33 t        2 [SPACE] 80 me      2 [SPACE] 136 te     84 ll
  2 [SPACE]   74 you      7 ,        2 [SPACE] 21 h       22 i       22 i
  2 [SPACE]   27 n       22 i        2 [SPACE] 32 s       44 er      22 i
206 ous        9 .
```

Three things worth noticing:

* **Sheng is handled by the base vocabulary.** `sasa`, `wewe`, `unajua`, `mambo`
  are spelled out of Latin character merges — the model does not need a new
  tokenizer entry to *say* them, only to *pronounce them in Sheng* (a T3 job).
* **English words cost more ids than Swahili** (`serious` -> `s er i ous`, four
  ids) because the merge table is English-ASCII-heavy; the tag tells the model
  which phonology to apply.
* 51 text ids for 69 characters = **1.35 chars/id**; a 30 s clip transcript
  (≈450 chars) fits comfortably inside `max_text_tokens = 2048`.

Then `mtl_tts.generate` wraps it exactly as the trainer does:

```python
text_tokens = tokenizer.text_to_tokens(text, language_id="sw")   # (1, 51)
text_tokens = torch.cat([text_tokens, text_tokens], dim=0)       # (2, 51)  <- CFG pair
text_tokens = F.pad(text_tokens, (1, 0), value=255)              # prepend [START]
text_tokens = F.pad(text_tokens, (0, 1), value=0)                # append  [STOP]
# -> (2, 53) int64
```

The `(2, …)` batch is not a bug: the second row is the **classifier-free
guidance unconditional branch**. It is identical here, and gets zeroed later by
`prepare_input_embeds` (`text_emb[1].zero_()`).

---

## 4. Stage B — T3: text tokens to speech tokens

T3 is `LlamaModel` with a custom embedding front-end, a custom conditioning
prefix, and **two** output heads. It never sees token ids through Llama's own
`embed_tokens`; the embeddings are injected as `inputs_embeds`.

```
T3
├── tfmr            LlamaModel (LLAMA_520M)
│     hidden 1024, 30 layers, 16 heads x head_dim 64, MLP 4096, SwiGLU
│     RoPE theta 500000, llama3 rope_scaling factor 8
│     vocab_size 8 -- unused, we bypass embed_tokens
├── text_emb        Embedding(2454, 1024)
├── speech_emb      Embedding(8194, 1024)      (6561 codes + specials, padded)
├── text_pos_emb    LearnedPositionEmbeddings(2050, 1024)
├── speech_pos_emb  LearnedPositionEmbeddings(4100, 1024)
├── cond_enc        T3CondEnc   (speaker + prompt + emotion -> prefix)
├── text_head       Linear(1024 -> 2454, no bias)
└── speech_head     Linear(1024 -> 8194, no bias)
```

Parameter count from those shapes: `30 x (attn 4x1024x1024 + mlp 3x1024x4096)`
≈ 520 M in the transformer, plus `1.0 M + 8.4 M` in the two embedding tables,
plus `2.5 M + 8.4 M` in the two heads, plus `0.26 M` in `cond_enc` → **≈ 541 M**
including both tables (`~1 GB` in fp16, which is why a checkpoint dir is ~1 GB).

### 4.1 The conditioning prefix — `T3CondEnc`

This is the only place "who is speaking" enters the LM. Three tensors, three
different jobs, one concatenated prefix of **34 positions**:

```
T3Cond.speaker_emb                (B, 256)      ve output, mean over windows
  └─ spkr_enc: Linear(256 -> 1024)              -> (B, 1, 1024)

T3Cond.cond_prompt_speech_tokens  (B, 150)      first 6 s of the reference clip
  └─ speech_emb(ids) + speech_pos_emb(0..149)   -> (B, 150, 1024)
  └─ perceiver (32 queries, 4 heads, 2 attn blocks) -> (B, 32, 1024)

T3Cond.emotion_adv                (B, 1, 1)     the "exaggeration" scalar, 0.5 default
  └─ emotion_adv_fc: Linear(1 -> 1024, no bias) -> (B, 1, 1024)

cond_emb = cat([spkr(1), clap(0), prompt(32), emotion(1)], dim=1)   -> (B, 34, 1024)
```

`Perceiver.forward` is `cross_attn(query(32), h(150))` followed by
self-attention over the 32 latents: it is a *learned compression* of the
reference clip's token stream, not a copy. This is why the T3 prompt is 150
tokens but only costs 32 LM positions, and why a 6 s reference is enough —
32 latents is all the acoustic context the LM ever gets.

`clap_emb` is reserved in the dataclass but asserted `None`; no CLAP conditioning
ships in the released weights.

### 4.2 Sequence assembly — `prepare_input_embeds`

```
positions:   0 ...... 33 | 34 ............. 34+L+1 | 34+L+2 ..... 34+L+S+2
content:     conditioned  | [START] text [STOP]     | [START] speech ... [STOP]
             prefix (34)  | text, L+2 ids            | speech, S+2 ids
embeddings:  cond_emb     | text_emb + text_pos_emb  | speech_emb + speech_pos_emb
```

Two independent position spaces on purpose: text positions always restart at 0,
speech positions always restart at 0, and the prefix itself carries no positional
component. That is what lets the same weights handle a 3-word prompt and a
40 s paragraph — and it is what makes the `[START]`-of-speech hand-off stable.

Shapes for inference (CFG batch of 2, 51 text ids, one initial speech token):

```
cond_emb             (2, 34, 1024)
text_emb             (2, 53, 1024)     text row 1 zeroed for CFG
speech_emb           (2,  1, 1024)     the [START] speech token (6561)
embeds               (2, 88, 1024)     = 34 + 53 + 1
bos_embed            (2,  1, 1024)     [START] again + speech_pos_emb(0)
inputs_embeds        (2, 89, 1024)     <- concatenated, §5.1 explains the duplicate
```

### 4.3 Training forward pass

`T3.forward` runs one causal pass over the whole concatenated sequence, then
**slices the hidden states back into two streams** and applies each head:

```
hidden_states            (B, 34+L+2+S+2, 1024)
text_latents   = hidden[:, 34        : 34+L+2]      (B, L+2, 1024) -> text_head   -> (B, L+2, 2454)
speech_latents = hidden[:, 34+L+2    : 34+L+2+S+2]  (B, S+2, 1024) -> speech_head -> (B, S+2, 8194)
```

`T3.loss` then computes **two cross-entropies**, one per stream, and returns them
separately:

```python
loss_text   = F.cross_entropy(text_logits,   masked_text,   ignore_index=-100)
loss_speech = F.cross_entropy(speech_logits, masked_speech, ignore_index=-100)
return loss_text, loss_speech
```

`train_t3.py` optimises the sum (`loss_text + loss_speech`). The text loss is a
regulariser: it forces the LM to keep the spelling of the transcript recoverable
in its own hidden states, which is what keeps text and speech aligned in a model
that has no explicit duration predictor.

> **Important caveat — read before training.** As shipped, this loss is *not*
> shifted, i.e. the state at position `p` is scored against the token at `p`.
> Generation scores the state at the last position against the *next* token
> (§5). The widely used community kit
> [`gokhaneraslan/chatterbox-finetuning`](https://github.com/gokhaneraslan/chatterbox-finetuning)
> implements the shifted variant (`speech_logits[:, :-1]` vs
> `speech_tokens[:, 1:]`) and also masks the reference-prompt region out of the
> speech labels. `docs/03-chatterbox-finetuning.md` gives the exact patch; the
> difference is `loss` moving from "reconstruct" to "predict", and it shows up as
> runaway or repetitive generation after a long fine-tune.

### 4.4 Dual heads, one backbone

Having two heads over one backbone is what makes the same 520 M network both a
text autoencoder and a speech generator:

* `text_head` never runs at inference — it only exists to create the gradient
  signal that pins text content into the shared hidden states;
* `speech_head` is the only thing sampled from at inference;
* both read the *same* final hidden states, so "where am I in the text" and
  "what sound comes next" are solved by one attention stack.

---

## 5. Stage B at inference — the autoregressive loop

`T3.inference` (`models/t3/t3.py`, ~line 227) is a hand-written loop, not
`generate()` (the `generate()` call is commented out in the source). It is worth
reading in full because every quality knob lives here.

### 5.1 Why `inputs_embeds` twice

```python
bos_token  = [[6561]]                     # start_speech_token
bos_embed  = speech_emb(bos_token) + speech_pos_emb.get_fixed_embedding(0)
inputs_embeds = cat([embeds, bos_embed], dim=1)   # embeds already ends with a speech token
```

`embeds` ends with the *conditioning-start* speech position (from
`prepare_input_embeds`), and one more BOS is appended so that the **last hidden
state is a pure "speech continues here" state**. Logits at that position give the
first real speech token. Generation is therefore always "predict token at
`len(inputs_embeds)`", which is exactly the position convention the trainer's
learned `speech_pos_emb` was indexed against (`get_fixed_embedding(i+1)` for the
`i`-th emitted token).

### 5.2 Guidance, warpers, and the alignment watchdog

Per step, in this exact order:

```
logits_step                     (2, 8194)         B=2: conditional + unconditional
cond, uncond = logits_step[0], logits_step[1]
logits = cond + cfg_weight * (cond - uncond)      (1, 8194)   default cfg_weight=0.5
        │  alignment_stream_analyzer.step(logits, last_token)   see 5.3
        │  RepetitionPenaltyLogitsProcessor(2.0)  [note: 2.0 > typical 1.1]
        │  temperature 0.8
        │  MinPLogitsWarper(0.05), TopPLogitsWarper(1.0)
        ▼
probs = softmax(logits) -> multinomial -> next_token (1, 1)
```

Two details that surprise people:

* **CFG is over the whole prefix, not just the speaker.** Row 1 of the batch has
  `text_emb[1].zero_()`, so the unconditional branch is "same conditioning, no
  text" — guidance sharpens text adherence, not timbre.
* The unconditional branch and the conditional branch share `cond_emb`
  (`cond_emb` is expanded from `(1, 34, 1024)` to `(2, 34, 1024)`), so guidance
  cannot drift the speaker away from the reference.

### 5.3 The multilingual integrity checker

For multilingual checkpoints only (`hp.is_multilingual`), `T3.inference` attaches
`AlignmentStreamAnalyzer(tfmr, alignment_layer_idx=9, eos_idx=6562)`. It hooks
attention on three **hand-picked heads** — `LLAMA_ALIGNED_HEADS = [(12,15),
(13,11), (9,2)]`, i.e. layers 12/13/9, heads 15/11/2 — and averages them into an
alignment matrix `A[t, s]` (speech frame x text token). From `A` it derives:

| flag | rule | effect |
|---|---|---|
| `false_start` | bottom-right activations or no strong early activation | hold generation, do not commit the first frames (anti-hallucination at the start) |
| `complete` | argmax text position reaches `S-3` | generation has consumed the text |
| `long_tail` | final-3-token mass ≥ 5 frames > 200 ms | force `[STOP]` speech token 6562 |
| `repetition` | activation returns to earlier text after completion | force `[STOP]` |
| `discontinuity` | text position jumps | flag only |

This is the mechanism behind the empirical rule in `finetune-chatterbox`'s
`EVAL_PROMPTS`: long texts "degrade" because the watchdog trips earlier, and
runaway generation is cut at 1000 tokens (`max_new_tokens=1000` in `mtl_tts.py`).

### 5.4 Stop conditions

Generation ends on the first of:

1. sampled `next_token == 6562` (`[STOP]` speech);
2. `max_new_tokens = 1000` (≈ 40 s of speech at 25 Hz);
3. an alignment-watchdog override that forces `[STOP]`.

### 5.5 Clean-up before the vocoder

```python
speech_tokens = speech_tokens[0]            # drop the CFG row
speech_tokens = drop_invalid_tokens(speech_tokens)   # remove ids >= 6561
```

`drop_invalid_tokens` (`models/s3tokenizer/__init__.py`) is essential: T3's
output vocabulary is 8194 (special tokens included), the vocoder's token
embedding only has 6561+1 rows. Any id ≥ 6561 left in would index out of range in
`flow.py` (`input_embedding`). The S3Gen forward path also has a defensive
`.any()` check that logs `out-of-range special tokens found in flow, fix inputs!`.

### 5.6 Output of T3

```
speech_tokens: (S,) int64 in [0, 6560]        # S = round(audio_seconds * 25)
```

Nothing else. No prosody vector, no durations, no pitch — those are all implicit
in the token sequence and re-derived by S3Gen.

---

## 6. Stage C — S3Gen part 1: tokens to mel

`S3Token2Mel` (`models/s3gen/s3gen.py`) is a CosyVoice-lineage pipeline. Three
separate things come out of the reference clip before any token is processed.

### 6.1 `embed_ref` — the s3gen half of voice cloning

```python
ref_24k  = librosa.load(wav, sr=24000)[:10 s]          # DEC_COND_LEN = 10 * 24000
ref_16k  = resample(ref_24k, 24000 -> 16000)           # and ref_16k[:6 s] for T3

prompt_feat    = mel_extractor(ref_24k)                # (1, 80, T_ref)   @ 50 fps
embedding      = CAMPPlus(ref_16k)                     # (1, 192)         x-vector
prompt_token   = S3Tokenizer(ref_16k)                  # (1, 25*T_ref)    @ 25 Hz
```

and the rate contract is enforced on the spot:

```python
if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
    ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
```

`prompt_feat` is the only tensor that survives into the CFM as *audio*; the
x-vector is projected to 80-d and injected as a per-frame bias; `prompt_token`
is replayed through the same encoder as the generated tokens so the encoder's
receptive field "warms up" on the reference.

### 6.2 Token -> encoder -> mel-rate features

```python
embedding = F.normalize(embedding); embedding = spk_embed_affine_layer(embedding)   # (1,192)->(1,80)
token     = cat([prompt_token, generated_token], dim=1)        # (1, T_prompt + S)
mask      = ~make_pad_mask(token_len).unsqueeze(-1)            # (1, T, 1)
x         = input_embedding(token) * mask                      # (1, T, 512)
h, h_masks = encoder(x, token_len)                             # UpsampleConformerEncoder
h         = h[:, :-pre_lookahead_len * token_mel_ratio]         # causal trim, 3*2 = 6 frames
h         = encoder_proj(h)                                    # (1, 2T, 80)   <- x2 time upsample
mu        = h.transpose(1, 2)                                  # (1, 80, 2T)
conds     = zeros(1, 80, mel_len1 + mel_len2); conds[:, :mel_len1] = prompt_feat
```

The encoder (`UpsampleConformerEncoder`, 6 blocks, 512-d, 8 heads, rel-pos
ESPNet style, `use_cnn_module=False`, `macaron_style=False`) contains the x2
time upsampling internally, so **one token in -> two mel frames out**, i.e.
25 Hz -> 50 fps. `encoder_proj` is `Linear(512 -> 80)`.

> Rate note: the s3gen mel grid is **50 fps** (`n_fft=1920`, `hop_size=480`,
> `fmax=8000` at 24 kHz) — not 100 fps. Two frames per token is exactly the 40 ms
> duration of one S3 token. HiFT then multiplies the frame rate by
> `8 x 5 x 3 = 120` and each output frame expands to `istft hop_len = 4` samples:
> `50 * 120 * 4 = 24000` samples/s. (A sibling walkthrough in
> `voice-clone/docs/` quotes 100 fps / hop 240; the shipped defaults above are
> what the weights were trained with.)

The two conditionings are now visible in one picture:

```
mu    (1, 80, 2*(T_prompt+S))    content: "what is said"        <- tokens
cond  (1, 80, 2*T_ref + 2*S)     texture: "what the voice is"   <- reference mel, in front
spks  (1, 80)                    identity: "who"                <- CAMPPlus x-vector
```

### 6.3 What the CFM actually solves

`CausalConditionalCFM` is **conditional flow matching**, not a diffusion model:

```
x0 ~ N(0, I)                                  # (1, 80, T_mel)
for t, r in zip(t_span[:-1], t_span[1:]):      # 10 steps, cosine schedule
    dxdt = estimator(x, mask, mu, t, spks, cond)
    x = x + (r - t) * dxdt
```

The estimator (`ConditionalDecoder`, causal conv + transformer hybrid) receives
`in_channels=320 = 80 (x) + 80 (mu) + 80 (spks broadcast) + 80 (cond mel)`, plus
a timestep embedding, and predicts an 80-d velocity per frame. `n_blocks=4`,
`num_mid_blocks=12`, `channels=[256]`, 8 heads, `gelu`, `pre_lookahead_len=3`.

Guidance in this half is **speaker-only**, and it is implemented by literally
zeroing `spks` for the mirrored batch half:

```python
x_in[:B] = x_in[B:] = x        # same noise
mu_in[:B] = mu                 # conditional branch gets content
spks_in[:B] = spks             # unconditional branch gets zeros
cond_in[:B] = cond             # reference mel in both halves
...
dxdt = (1 + 0.7) * dxdt_cond - 0.7 * dxdt_uncond     # inference_cfg_rate = 0.7
```

`CFM_PARAMS`: `sigma_min 1e-6`, `solver euler`, `t_scheduler cosine`,
`training_cfg_rate 0.2`, `inference_cfg_rate 0.7`, `n_timesteps 10`
(`2` in the `meanflow` variants, which is what `tts_turbo.py` uses).

Finally the prompt frames are cut back off:

```python
feat = feat[:, :, mel_len1:]        # keep only the generated region
assert feat.shape[2] == mel_len2
```

---

## 7. Stage D — mel to 24 kHz waveform (HiFT)

`HiFTGenerator` (`models/s3gen/hifigan.py`) is a HiFi-GAN with a pitch source.

```
mel (1, 80, T) @ 50 fps
 ├── ConvRNNF0Predictor -> F0 (1, T)  ->  sine source at harmonic frequencies,
 │                                      upsampled by prod(upsample_rates) * hop_len = 480
 └── upsample_rates [8, 5, 3] = x120  ->  50 * 120 = 6000 frames/s
      followed by 3 fused ResBlock banks (kernel 3/7/11, dilations 1/3/5)
      + source fusion -> final conv -> torch.istft(n_fft=16, hop_len=4)
      -> 6000 * 4 = 24000 samples/s -> waveform (1, N)
```

The F0 predictor is what makes it HiFT rather than plain HiFi-GAN: the excitation
is a harmonic source driven by predicted pitch, so a cloned voice keeps its
natural pitch instead of a learned buzz. `n_fft=16, hop=4` for the F0 branch
(`istft_params`), `num_kernels = 3`.

Then, before returning, `S3Gen.inference` applies the anti-spillover fade:

```python
output_wavs[:, :len(trim_fade)] *= trim_fade     # 20 ms cosine fade-in
```

That is a deliberate, documented hack: the first syllable of the reference clip
otherwise bleeds into the output. It is also why an over-short reference clip
sounds like the clone "starts mid-word" — the fade eats the onset.

---

## 8. Stage E — watermarking

`ChatterboxMultilingualTTS.__init__` constructs
`perth.PerthImplicitWatermarker()` and `generate` always applies it:

```python
wav = wav.squeeze(0).float().numpy()                  # 24 kHz
watermarked = watermarker.apply_watermark(wav, sample_rate=self.sr)
return torch.from_numpy(watermarked).unsqueeze(0)      # (1, N) float32
```

`perth` (Resemble's Perth neural watermarker) resamples internally to 48 kHz,
embeds an imperceptible watermark, and returns the same length at the original
rate. Consequences for this repo: **exported voices are watermarked**, and any
audio-metric comparison must compare like with like (the baseline is watermarked
too, so CER/speaker-similarity deltas stay meaningful).

---

## 9. One full trace, all shapes

Example sentence: `Sasa wewe, unajua vile mambo iko. Let me tell you, hii ni serious.`
51 text ids, ≈2.9 s of audio -> 72 S3 tokens. Reference clip: 6 s.

| # | tensor | shape | dtype | where |
|---|---|---|---|---|
| 1 | text ids | `(2, 53)` | int64 | after `[START]`/`[STOP]` + CFG duplicate |
| 2 | speaker_emb | `(1, 256)` | f32 | `ve.embeds_from_wavs` |
| 3 | cond_prompt_speech_tokens | `(1, 150)` | int64 | S3 tokens of first 6 s |
| 4 | emotion_adv | `(1, 1, 1)` | f32 | `exaggeration = 0.5` |
| 5 | cond_emb | `(1, 34, 1024)` | bf16 | `T3CondEnc` |
| 6 | text_emb | `(2, 53, 1024)` | bf16 | row 1 zeroed for CFG |
| 7 | speech_emb (BOS) | `(2, 1, 1024)` | bf16 | `[START]` speech |
| 8 | inputs_embeds | `(2, 89, 1024)` | bf16 | first forward |
| 9 | logits (last pos) | `(2, 8194)` | f32 | `speech_head` |
| 10 | sampled token | `(1, 1)` | int64 | after CFG + penalties |
| 11 | KV cache per step | 30 layers x `(2, 16, T, 64)` K and V | bf16 | grows by 1 position per token |
| 12 | speech_tokens | `(72,)` | int64 | after `drop_invalid_tokens` |
| 13 | prompt_token | `(1, 150)` | int64 | 6 s @ 25 Hz |
| 14 | flow token seq | `(1, 222)` | int64 | 150 prompt + 72 generated |
| 15 | encoder out `h` | `(1, 444, 512)` | bf16 | 2 x (150 + 72) frames @ 50 fps |
| 16 | `mu` | `(1, 80, 444)` | bf16 | after `encoder_proj` |
| 17 | `cond` | `(1, 80, 444)` | bf16 | first 300 frames = reference mel, rest zero |
| 18 | CFM batch | `(2, 80, 444)` | bf16 | CFG over speaker |
| 19 | generated mel | `(1, 80, 144)` | f32 | 444 - 300 = 2 x 72 frames |
| 20 | waveform | `(1, 69 120)` | f32 | 144 frames x 480 samples = 2.88 s |
| 21 | watermarked wav | `(1, 69 120)` | f32 | returned |

Note on row 17: `conds` is allocated for `mel_len1 + mel_len2` frames, where
`mel_len1 = prompt_feat.shape[1]` — the reference mel *defines* the length of the
conditioning block — and `mel_len2 = h.shape[1] - mel_len1`. A 6 s reference at
50 fps gives 300 frames = 2 x 150 prompt tokens, which is exactly why
`embed_ref` asserts `mel_len == 2 * token_len` and truncates the tokens if the
reference was not padded to a 40 ms multiple.

---

## 10. Where the cost goes

For a ~3 s output on a Colab L4 (bf16):

| stage | calls | share of wall time | notes |
|---|---|---|---|
| T3 prefill (89 positions) | 1 | ~2 % | one 520 M pass |
| T3 decode | 72 | **~55 %** | one 520 M forward per token, KV-cached |
| S3 encoder | 1 | ~3 % | 6 conformer blocks over 444 frames |
| CFM Euler | 10 | **~30 %** | 10 velocity-net calls |
| HiFT | 1 | ~10 % | conv-heavy, x480 expansion |
| Perth watermark | 1 | ~2 % | inside `generate` |

Two levers that exist in the source but are not in the public API docs:

* `s3gen.inference(..., n_cfm_timesteps=16)` buys mel quality without retraining;
  10 is the default, 2 is the `meanflow` (turbo) path;
* T3 decode is strictly per-token, so **output length dominates latency**; the
  reference costs a constant 34 prefix positions regardless of its length.

Measured RTF for this project's evals is ~0.15-0.35 on an L4 with watermarking
included (see the `rtf` field in `evals/<run>/eval.json`).

---

## 11. Invariants and what breaks where

These are the checks that make debugging tractable, and each maps onto a symptom
the `voice-clone` scorer already labels:

| invariant | broken -> symptom | root cause |
|---|---|---|
| one `[START]` and one `[STOP]` in text ids | assertion in `_ensure_BOT_EOT` | malformed prompt |
| every speech id `< 6561` | `input_embedding` index error | special tokens leaked past `drop_invalid_tokens` |
| `mel_len == 2 * token_len` | warning, then silent truncation | reference not padded to 40 ms multiples |
| T3 emits 6562 before 1000 tokens | `runaway` (40 s of babble) | over-trained / LR too high |
| CFM does not collapse | `silent` (RMS < 1e-3) | wrong language id, mismatched T3 |
| F0 source sane | `clipping` (>1 % of samples at full scale) | degenerate token stream into the vocoder |
| speaker x-vector present | `wrong-voice` (cosine < 0.5) | reference clip from the wrong speaker, or T3 overfit |

The architectural payoff: `silent`/`clipping` cannot be caused by a bad T3
fine-tune (s3gen is frozen), and `runaway` cannot be caused by s3gen. That is
precisely the premise of the measure-first design in `voice-clone`.

---

## 12. Mapping repo code to stages

| repo symbol | stage | file |
|---|---|---|
| `Featurizer.text_tokens` | A | `finetune-chatterbox/src/sheng_chatterbox/t3data.py` |
| `Featurizer.speech_tokens` / `cond_tokens` | C (tokenizer only) | same |
| `Featurizer.speaker_embedding` | B (`ve`) | same |
| `T3Cond(...)` + `model.t3.loss` | B / training | `train_t3.py::batch_loss` |
| `SpeechCollator` | B (padding + lens) | `t3data.py` |
| `_generate` | A -> E | `infer.py` |
| `load_finetuned` | swaps only T3 | `infer.py` |
| `export` | copies base files, replaces T3 | `infer.py` |
| `evals.synthesize` | A -> E + ASR judge | `voice-clone/src/sheng_voiceclone/evals.py` |
| `s3gen-cfm-walkthrough.md` | C/D in more depth | `voice-clone/docs/` |

`export` is worth stressing: because the split is clean, packaging a fine-tune is
"copy `ve.pt`, `s3gen.pt`, the tokenizer JSON and `conds.pt`, then drop the new
`t3_cfg.safetensors` next to them". No conversion, no merge, no custom loader:

```python
ChatterboxMultilingualTTS.from_local(dir, device, t3_model="t3_cfg.safetensors")
```

(`from_local` without `t3_model=` is the PyPI 0.1.7 signature and hard-codes
`t3_mtl23ls_v2.safetensors`; master added the argument, which is why
`sheng_chatterbox` pins the filename explicitly.)

---

## 13. Appendix: defaults that matter

```python
# T3Config (multilingual)
start_text_token   = 255      stop_text_token   = 0
start_speech_token = 6561     stop_speech_token = 6562
text_tokens_dict_size = 2454  speech_tokens_dict_size = 8194
max_text_tokens = 2048        max_speech_tokens = 4096
speech_cond_prompt_len = 150  speaker_embed_size = 256
use_perceiver_resampler = True   emotion_adv = True   input_pos_emb = "learned"

# generate() defaults (mtl_tts.py)
exaggeration = 0.5   cfg_weight = 0.5   temperature = 0.8
repetition_penalty = 2.0   min_p = 0.05   top_p = 1.0   max_new_tokens = 1000

# S3Gen
ENC_COND_LEN = 6 s @ 16 kHz      DEC_COND_LEN = 10 s @ 24 kHz
S3_SR = 16000   S3GEN_SR = 24000   token_mel_ratio = 2
n_cfm_timesteps = 10 (2 meanflow)   inference_cfg_rate = 0.7
mel: n_fft 1920, hop 480, 80 mels, fmax 8000     -> 50 fps
HiFT: upsample [8, 5, 3] x istft hop 4 = x480
```
