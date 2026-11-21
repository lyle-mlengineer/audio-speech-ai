# How S3Gen + its CFM decoder work (Chatterbox)

Reference walkthrough of the **frozen** half of Chatterbox: the part our
`voice-clone` fine-tune never touches. Source read from
`site-packages/chatterbox/models/s3gen/` (package `chatterbox-tts`, CosyVoice
heritage, Apache-2.0).

Line refs are to `chatterbox/models/s3gen/` unless stated otherwise.

---

## Where it sits

```
T3 (what we train)          S3Gen (frozen)
─────────────────           ──────────────────────────────────────────────
text + speaker ──► tokens ──► S3Token2Mel: tokens ──► CFM ──► mel (80 band, 100 fps)
                (25 Hz, VQ)                    └─► HiFT: mel ──► 24 kHz waveform
```

`S3Token2Wav.inference()` (`s3gen.py:325`) is the whole thing:
`flow_inference()` -> mel, `hift_inference()` -> waveform.

Rate contract:

| quantity | value | why |
|---|---|---|
| S3 tokens | 25 Hz (40 ms each) | speech tokenizer output |
| mel frames | 100 fps (10 ms, hop 240 @ 24 kHz) | `S3GEN_SR = 24000` (`const.py`) |
| ratio | **2 mel frames per token** | hardcoded `token_mel_ratio=2` (`flow.py:52`) |
| waveform | 24 kHz | HiFT upsample `8 x 5 x 3 = 120` |

---

## Step 0 - three things are pulled from the reference clip

`embed_ref()` (`s3gen.py:118-175`) returns the dict that conditions everything
downstream:

| output | from | shape | used by |
|---|---|---|---|
| `prompt_feat` | 24 kHz reference -> 80-band mel | `(1, T_ref, 80)` | CFM `cond` ("continue this audio") |
| `embedding` | 16 kHz reference -> **CAMPPlus** x-vector | `(1, 192)` | projected to 80-d, conditions the velocity net |
| `prompt_token` | 16 kHz reference -> S3 tokenizer | `(1, 25*T_ref)` | prefix tokens replayed through the encoder |

The rate contract is enforced right there (`s3gen.py:157-164`): the mel length is
forced to `2 * token_length`, and the tokens are truncated if it is not.

At inference the reference window is capped: `ENC_COND_LEN = 6 s` for T3,
`DEC_COND_LEN = 10 s` for s3gen (`mtl_tts.py`). A longer reference is a warning,
not an error - the tail is simply unused.

---

## Step 1 - token -> mel (`flow.py::CausalMaskedDiffWithXvec.inference`)

```python
token = concat([prompt_token, generated_token])      # prefix + new
mask  = ~make_pad_mask(token_len)
h, h_masks = self.encoder(token, token_len)         # UpsampleConformerEncoder
h  = self.encoder_proj(h)                           # 512 -> 80
mu = h.transpose(1, 2)                              # (B, 80, T_mel)
conds[:, :mel_len1] = prompt_feat                   # reference mel in front
embedding = spk_embed_affine_layer(F.normalize(xvec))   # -> 80
```

The encoder is a 6-block rel-pos ESPnet conformer (`s3gen.py:74-90`) that does the
`x2` time upsampling internally, so it emits one 80-d frame per 10 ms. The
reference mel frames are cut off the output at the end, so the result is exactly
`2 * n_new_tokens` frames.

---

## Step 2 - the CFM: an ODE solved with Euler

Not a DDPM. It is **conditional flow matching**: learn a velocity field, then
integrate noise -> data.

`solve_euler()` (`flow_matching.py:78-145`):

```python
t_span = torch.linspace(0, 1, n_timesteps + 1)      # 10 steps by default
for t, r in zip(t_span[:-1], t_span[1:]):
    dxdt = estimator(x, mask, mu, t, spks, cond)     # velocity prediction
    x = x + (r - t) * dxdt                           # one Euler step
```

Two details that matter in practice:

**a) Classifier-free guidance is over the speaker vector only.** The batch is
doubled, and `cond_in[:B] = cond_in[B:] = cond` (shared) while `spks_in[:B] = spks`
and the second half is left at zeros. So the unconditional branch is "same
content, no speaker":

```python
dxdt = (1 + 0.7) * dxdt_cond - 0.7 * dxdt_uncond     # inference_cfg_rate = 0.7
```

Raising it strengthens speaker identity at the cost of brittle prosody and more
reference-clip leakage.

**b) Step count is the quality/cost dial.** `flow_inference()` (`s3gen.py:288`):
`n_cfm_timesteps or 10`, or `2` in meanflow mode. That is 10 velocity-net calls
per clip - the dominant synthesis cost.

`configs.py` (`CFM_PARAMS`): `sigma_min 1e-6`, `solver euler`, `t_scheduler cosine`,
`training_cfg_rate 0.2`, `inference_cfg_rate 0.7`.

---

## Step 3 - the velocity estimator (`decoder.py::ConditionalDecoder`)

Causal conv + transformer hybrid, 320 in / 80 out, `channels=[256]`, `n_blocks=4`,
`num_mid_blocks=12`, 8 heads, `act_fn='gelu'` (`s3gen.py:92-105`). Per call
(`decoder.py:243-274`):

```python
x = pack([x, mu], "b * t")      # noisy mel + encoder output
x = pack([x, spks], "b * t")    # + broadcast speaker vector
...                             # + t embedding, + cond mel
```

At every Euler step the net sees: the current noisy mel, `mu` (content from text
tokens), the speaker, the timestep, and the reference mel.

Because it is causal with `pre_lookahead_len=3` (`flow.py:53`), streaming is
possible, but in non-`finalize` mode the last `3 * 2 = 6` mel frames are trimmed
(`h[:, :-pre_lookahead_len * token_mel_ratio]`).

---

## Step 4 - mel -> waveform (`hifigan.py::HiFTGenerator`, `s3gen.py:262-292`)

```
mel 80-d @100 fps --> ConvRNNF0Predictor --> F0 --> sine/harmonic source
        \--> upsample x8 x5 x3 (=120) --> 24 kHz --> resblocks + source fusion --> wav
```

`upsample_rates=[8,5,3]` gives 100 * 120 = 24 kHz exactly. The F0 predictor is what
makes it a **HiFT** rather than a plain HiFi-GAN: pitch drives the harmonic source,
so the voice keeps its natural pitch instead of a learned excitation.

Then a 20 ms cosine fade-in is applied to the first samples (`s3gen.py:283-292`,
applied at `:352` and `:356`) - explicitly to suppress "spillover", i.e. the
reference clip's first syllable bleeding into the output. That fade is why an
over-short reference clip sounds like the clone starts mid-word.

---

## Step 5 - how the frozen half was trained (contrast with T3)

`compute_loss()` (`flow_matching.py:147-187`):

```python
t = rand(); if cosine: t = 1 - cos(t * pi / 2)
z = randn_like(x1)
y = (1 - (1 - sigma_min) * t) * z + t * x1        # interpolate noise -> data
u = x1 - (1 - sigma_min) * z                      # the true velocity
if training_cfg_rate > 0:                         # 20% of batches: drop all conditioning
    mu, spks, cond = mu * cfg_mask, spks * cfg_mask, cond * cfg_mask
pred = estimator(y, mask, mu, t, spks, cond)
loss = mse(pred * mask, u * mask) / (sum(mask) * 80)
```

* Despite `reg_loss_type: "l1"` in `CFM_PARAMS`, the implementation is **MSE**.
* The 20% conditioning dropout is what makes the zeroed-speaker branch meaningful
  at inference; without it CFG would be extrapolating from an untrained network.

---

## Why this matters for the Sheng pipeline

1. **Our fine-tune never touches s3gen** (`finetune-chatterbox/src/sheng_chatterbox/train_t3.py:103-106`
   freezes `ve` and `s3gen`). Waveform quality is fixed; only *which voice* the
   tokens encode changes. Noise in the output is a token problem, not a vocoder
   problem.
2. **`sheng-voiceclone`'s waveform verdicts map onto this pipeline**: `silent` =
   CFM collapsed, `clipping` = vocoder/F0 blowup after bad tokens, `runaway` = T3
   never emitted EOS (nothing to do with the decoder).
3. **The reference clip is used twice** - 6 s through `ve` for T3, up to 10 s
   through s3gen for `prompt_feat` + x-vector. That is why the `data` stage picks
   the *longest* validation clip.
4. **10 Euler steps is a floor, not a rule.** If a checkpoint sounds slightly
   degraded, `--n-cfm-timesteps 16` sometimes rescues it with no retraining.
