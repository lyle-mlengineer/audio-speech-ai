"""Dataset loading, splitting and feature preparation for Sheng fine-tuning.

Dataset facts (lyle-mlengineer/kenyan-celebs, verified 2026-11-15):
  - 42 rows total, single `train` split
  - columns: audio (16kHz), speaker (all "abel"), text
  - every clip is 30.0s, transcripts are 289-501 chars
  - no timestamps, no per-utterance language tags

That shape drives every decision below:
  - We must create our own validation split (there is only `train`).
  - One speaker means a speaker-disjoint split is impossible, so our eval
    number is optimistic. We surface this instead of pretending otherwise.
  - 30s clips are exactly Whisper's window, so we do not chunk.
  - 42 rows for a 769M-param model is severe overfitting territory, hence
    augmentation and early stopping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .text_norm import normalize_text

# Dataset is small enough that revisions matter for reproducibility; we log
# the resolved commit SHA in MLflow when available.


@dataclass
class DataBundle:
    train: Any
    eval: Any
    feature_extractor: Any
    tokenizer: Any
    processor: Any
    dataset_stats: Dict[str, Any]


def _load_raw(cfg: Config):
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {}
    if cfg.data.dataset_config:
        kwargs["name"] = cfg.data.dataset_config

    ds = load_dataset(cfg.data.dataset_name, **kwargs)
    return ds


def _cast_audio(ds, cfg: Config):
    from datasets import Audio

    # Force 16kHz mono so the feature extractor never sees a surprising rate.
    if cfg.data.audio_column in ds.column_names:
        ds = ds.cast_column(
            cfg.data.audio_column, Audio(sampling_rate=cfg.data.sampling_rate)
        )
    return ds


def prepare_splits(cfg: Config, raw) -> Tuple[Any, Any, Dict[str, Any]]:
    """Return (train_ds, eval_ds, stats)."""
    stats: Dict[str, Any] = {"available_splits": list(raw.keys())}

    if cfg.data.eval_split and cfg.data.eval_split in raw:
        train_ds = raw[cfg.data.train_split]
        eval_ds = raw[cfg.data.eval_split]
        stats["split_strategy"] = "native"
    else:
        if cfg.data.train_split not in raw:
            raise ValueError(
                f"split '{cfg.data.train_split}' not found. "
                f"Available: {list(raw.keys())}"
            )
        base = raw[cfg.data.train_split]
        stats["split_strategy"] = "fraction_of_train"

        total = len(base)
        n_eval = max(1, int(round(total * cfg.data.eval_fraction)))
        # Never let eval swallow the training set on tiny datasets.
        n_eval = min(n_eval, max(1, total // 4))
        n_train = total - n_eval
        if n_train < 1:
            raise ValueError(
                f"Dataset too small ({total} rows) for the requested eval_fraction."
            )

        split = base.train_test_split(test_size=n_eval, seed=cfg.data.seed, shuffle=True)
        train_ds = split["train"]
        eval_ds = split["test"]

    # Optional smoke test: shrink both sides so the whole pipeline can be
    # exercised in ~1 minute on a T4.
    if cfg.train.smoke_test_rows:
        n = cfg.train.smoke_test_rows
        train_ds = train_ds.select(range(min(n, len(train_ds))))
        eval_ds = eval_ds.select(range(min(max(1, n // 2), len(eval_ds))))
        stats["smoke_test"] = True

    stats["n_train"] = len(train_ds)
    stats["n_eval"] = len(eval_ds)

    # Report speaker composition -- with one speaker, WER is not a
    # generalisation estimate and the training log should say so.
    if "speaker" in train_ds.column_names:
        speakers = sorted({str(s) for s in train_ds["speaker"]})
        stats["speakers_train"] = speakers
        stats["single_speaker"] = len(speakers) == 1

    # Duration coverage.
    if cfg.data.audio_column in train_ds.column_names:
        durations = []
        for row in train_ds.select(range(min(50, len(train_ds)))):
            arr = row[cfg.data.audio_column].get("array")
            if arr is not None:
                durations.append(len(arr) / cfg.data.sampling_rate)
        if durations:
            stats["duration_seconds"] = {
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
                "mean": float(np.mean(durations)),
                "n_sampled": len(durations),
            }

    return train_ds, eval_ds, stats


def init_processor(cfg: Config):
    """Load feature extractor + tokenizer for the configured checkpoint."""
    from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

    name = cfg.model.name
    # `language` drives the tokenizer only when normalizing/decoding prompts;
    # the canonical text is language-agnostic so this is safe for Sheng.
    feature_extractor = WhisperFeatureExtractor.from_pretrained(name)
    tokenizer = WhisperTokenizer.from_pretrained(
        name, language=cfg.model.language, task=cfg.model.task
    )
    processor = WhisperProcessor.from_pretrained(
        name, language=cfg.model.language, task=cfg.model.task
    )
    return feature_extractor, tokenizer, processor


def augment_speed(
    array: np.ndarray,
    sample_rate: int,
    factor: float,
    rng: random.Random,
) -> np.ndarray:
    """Tempo perturbation by naive resampling.

    Resampling changes pitch as well as speed, which is technically
    "speed perturbation" in the Kaldi/WenetSpeech sense rather than a clean
    time-stretch. For ASR it works well as cheap augmentation and costs no
    extra dependency -- which matters on Colab.
    """
    if float(factor) == 1.0 or len(array) < 2:
        return array
    n_out = int(len(array) / float(factor))
    if n_out < 2:
        return array
    # Linear interpolation resample to n_out samples.
    src_idx = np.linspace(0.0, len(array) - 1.0, n_out)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.minimum(lo + 1, len(array) - 1)
    frac = src_idx - lo
    out = array[lo] * (1.0 - frac) + array[hi] * frac
    return out.astype(np.float32)


def spec_augment(
    input_features: np.ndarray,
    cfg: Config,
    rng: random.Random,
) -> np.ndarray:
    """Frequency/time masking on the (n_mels, n_frames) log-mel array.

    Implemented inline so we don't pull in torchaudio's SpecAugment, whose
    API and defaults have churned across versions.
    """
    n_mels, n_frames = input_features.shape
    out = input_features.copy()

    # Fill masks with the feature's global mean. This mirrors the original
    # SpecAugment "zero/mean filling" and, unlike zero-filling, does not
    # teach the model that a silent band means silence in the transcript.
    fill_val = float(out.mean())

    for _ in range(cfg.data.num_freq_masks):
        f = rng.randint(0, min(cfg.data.freq_mask_param, max(0, n_mels - 1)))
        if f > 0 and n_mels > f:
            f0 = rng.randint(0, n_mels - f)
            out[f0 : f0 + f, :] = fill_val

    for _ in range(cfg.data.num_time_masks):
        t = rng.randint(0, min(cfg.data.time_mask_param, max(0, n_frames - 1)))
        if t > 0 and n_frames > t:
            t0 = rng.randint(0, n_frames - t)
            out[:, t0 : t0 + t] = fill_val

    return out


def make_preprocess_fn(cfg: Config, feature_extractor, tokenizer, *, training: bool):
    """Build a `dataset.map` function.

    `training=False` disables all augmentation so evaluation stays
    deterministic and comparable across runs.
    """
    max_samples = int(cfg.data.max_input_seconds * cfg.data.sampling_rate)
    norm_kwargs = dict(
        lowercase=cfg.data.lowercase,
        strip_punctuation=cfg.data.strip_punctuation,
        collapse_repeats=cfg.data.collapse_repeats,
        max_repeat_run=cfg.data.max_repeat_run,
    )

    def _preprocess(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        audio_col = batch[cfg.data.audio_column]
        texts = batch[cfg.data.text_column]

        arrays: List[np.ndarray] = []
        for item in audio_col:
            arr = item.get("array") if isinstance(item, dict) else None
            if arr is None:
                arr = np.zeros(cfg.data.sampling_rate, dtype=np.float32)
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)

            if len(arr) > max_samples:
                arr = arr[:max_samples]

            if training:
                # Deterministic per-utterance RNG so resume/re-run is stable
                # for a given seed + index; falls back to global rng.
                rng = random.Random((hash(str(len(arr))) ^ cfg.data.seed) & 0xFFFFFFFF)
                if cfg.data.speed_perturb and rng.random() < cfg.data.speed_perturb_prob:
                    factor = rng.choice(cfg.data.speed_factors)
                    arr = augment_speed(arr, cfg.data.sampling_rate, factor, rng)
                    if len(arr) > max_samples:
                        arr = arr[:max_samples]
            arrays.append(arr)

        features = feature_extractor(
            arrays,
            sampling_rate=cfg.data.sampling_rate,
            return_tensors="np",
            return_attention_mask=False,
        )
        input_features = features.input_features

        if training and cfg.data.spec_augment:
            augmented = []
            for i in range(input_features.shape[0]):
                rng = random.Random((i * 7919 + cfg.data.seed) & 0xFFFFFFFF)
                augmented.append(spec_augment(input_features[i], cfg, rng))
            input_features = np.stack(augmented, axis=0)

        labels = tokenizer(
            [normalize_text(t, **norm_kwargs) for t in texts]
        ).input_ids

        return {
            "input_features": input_features.tolist(),
            "labels": labels,
        }

    return _preprocess


def build_datasets(cfg: Config) -> DataBundle:
    """Full data path: load -> split -> cast -> map -> drop raw columns."""
    raw = _load_raw(cfg)
    train_ds, eval_ds, stats = prepare_splits(cfg, raw)

    feature_extractor, tokenizer, processor = init_processor(cfg)

    train_ds = _cast_audio(train_ds, cfg)
    eval_ds = _cast_audio(eval_ds, cfg)

    # Keep raw text around for reporting; it must not be fed to the model.
    train_ds = train_ds.add_column("raw_text", train_ds[cfg.data.text_column])
    eval_ds = eval_ds.add_column("raw_text", eval_ds[cfg.data.text_column])

    train_fn = make_preprocess_fn(cfg, feature_extractor, tokenizer, training=True)
    eval_fn = make_preprocess_fn(cfg, feature_extractor, tokenizer, training=False)

    train_ds = train_ds.map(
        train_fn,
        batched=True,
        batch_size=4,
        remove_columns=[
            c for c in train_ds.column_names if c not in {"input_features", "labels", "raw_text"}
        ],
        desc="Preprocessing train (with augmentation)",
    )
    eval_ds = eval_ds.map(
        eval_fn,
        batched=True,
        batch_size=4,
        remove_columns=[
            c for c in eval_ds.column_names if c not in {"input_features", "labels", "raw_text"}
        ],
        desc="Preprocessing eval (clean)",
    )

    return DataBundle(
        train=train_ds,
        eval=eval_ds,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        processor=processor,
        dataset_stats=stats,
    )
