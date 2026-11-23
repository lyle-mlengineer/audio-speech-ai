# Chatterbox documentation set

Four documents on `ResembleAI/chatterbox` (multilingual), read from the shipped
`chatterbox-tts` source plus this repository's `finetune-chatterbox` and
`voice-clone`.

| # | document | what it answers |
|---|---|---|
| 1 | [`01-chatterbox-multilingual-architecture.md`](01-chatterbox-multilingual-architecture.md) | What are the parts, and how do tensors flow from text to 24 kHz audio? (stage-by-stage shapes, the T3 sequence layout, CFM/HiFT, one fully traced example) |
| 2 | [`02-chatterbox-training-recipe.md`](02-chatterbox-training-recipe.md) | How was the model trained upstream (reconstructed, tagged verified/inferred/unknown), and exactly how is the Sheng fine-tune configured? |
| 3 | [`03-chatterbox-finetuning.md`](03-chatterbox-finetuning.md) | Full T3 fine-tune vs LoRA vs component-wise (text side / audio side / s3gen), with code, hyperparameters and guard rails |
| 4 | [`04-chatterbox-voice-cloning.md`](04-chatterbox-voice-cloning.md) | How zero-shot voice cloning works: the five tensors taken from the reference, where each enters, the knobs, and how to judge a clone |

PDFs are generated from the markdown:

```bash
python docs/build_pdfs.py          # all documents
python docs/build_pdfs.py 03       # only 03-*
```

`build_pdfs.py` renders markdown -> styled HTML -> PDF with headless
Chrome/Edge. The intermediate `.html` files are written next to the markdown.

Conventions used throughout:

* `code font` always means a real identifier or filename you can grep;
* claims about upstream training are tagged **[verified]**, **[inferred]** or
  **[unknown]** — no invented hyperparameters;
* rate contract: text ids -> S3 tokens at 25 Hz -> mel at 50 fps -> 24 kHz audio.
