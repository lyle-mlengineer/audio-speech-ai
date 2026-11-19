"""Stage 4: turn transcripts + WAVs into the ``audio_codes`` the trainer needs.

Qwen3-TTS is a discrete multi-codebook LM: 12 Hz, 16 codebooks. The upstream
script is ``finetuning/prepare_data.py``; this is the same thing with batching
that survives 30 s clips (the upstream default of 32 utterances per call is
~2 GB of activations for this dataset) and a progress line, because on Colab
you want to know whether the cell died or is just quiet.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .paths import Paths

BATCH = 8


def load_tokenizer(paths: Paths, device: str = "cuda:0"):
    from qwen_tts import Qwen3TTSTokenizer

    if not (paths.tokenizer_dir / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"{paths.tokenizer_dir} is empty - run `sheng-qwentts model` first"
        )
    return Qwen3TTSTokenizer.from_pretrained(str(paths.tokenizer_dir), device_map=device)


def add_codes(
    in_jsonl: Path,
    out_jsonl: Path,
    tokenizer,
    *,
    batch: int = BATCH,
    force: bool = False,
) -> int:
    """Write ``in_jsonl`` + ``audio_codes`` to ``out_jsonl``; returns line count."""
    if out_jsonl.is_file() and not force:
        done = sum(1 for _ in out_jsonl.open(encoding="utf-8"))
        total = sum(1 for _ in in_jsonl.open(encoding="utf-8"))
        if done == total and done > 0:
            return done

    lines = [json.loads(line) for line in in_jsonl.open(encoding="utf-8") if line.strip()]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    t0 = time.time()
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for start in range(0, len(lines), batch):
            chunk = lines[start : start + batch]
            enc = tokenizer.encode([row["audio"] for row in chunk])
            codes = enc.audio_codes
            if len(codes) != len(chunk):
                raise RuntimeError(f"tokenizer returned {len(codes)} codes for {len(chunk)} clips")
            for row, code in zip(chunk, codes):
                row["audio_codes"] = code.cpu().tolist()
                row["frames"] = int(code.shape[0])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            print(
                f"[codes] {written}/{len(lines)} clips "
                f"({time.time() - t0:.1f}s, last {codes[0].shape[0]} frames)",
                flush=True,
            )
    return written


def run(paths: Paths, *, device: str = "cuda:0", batch: int = BATCH, force: bool = False) -> dict:
    """Stage entry point: codes for train (and val, when a val split exists)."""
    tokenizer = load_tokenizer(paths, device=device)
    out = {"train": add_codes(paths.train_raw_jsonl, paths.train_codes_jsonl, tokenizer,
                              batch=batch, force=force)}
    if paths.val_raw_jsonl.is_file():
        out["val"] = add_codes(paths.val_raw_jsonl, paths.val_codes_jsonl, tokenizer,
                               batch=batch, force=force)
    frames = []
    for path in (paths.train_codes_jsonl, paths.val_codes_jsonl):
        if path.is_file():
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                frames.append(row.get("frames", len(row["audio_codes"])))
    out["frames_mean"] = round(sum(frames) / len(frames), 1) if frames else 0
    out["frames_max"] = max(frames) if frames else 0
    return out
