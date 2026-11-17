"""Model loading, training and evaluation for Whisper on Sheng.

Handles the transformers 4.x -> 5.x API drift explicitly, because Colab now
ships transformers 5.x where several argument names changed:
  - `evaluation_strategy`  -> `eval_strategy`
  - `tokenizer=`           -> `processing_class=`
  - `overwrite_output_dir` -> removed
Both spellings are supported here so the same code runs on local 4.x and
Colab 5.x without edits.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .data import DataBundle
from .metrics import compute_metrics


# --------------------------------------------------------------------- model


def load_model(cfg: Config):
    """Load WhisperForConditionalGeneration configured for transcription."""
    import torch
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(cfg.model.name)

    # Never train the generation config, but do pin it so every eval/generate
    # call uses the same decoding setup.
    model.generation_config.language = f"<|{_whisper_lang_code(cfg.model.language)}|>"
    model.generation_config.task = cfg.model.task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.return_timestamps = cfg.model.return_timestamps
    model.generation_config.suppress_tokens = []

    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


_LANG_ALIASES = {
    "swahili": "sw",
    "sw": "sw",
    "english": "en",
    "en": "en",
}


def _whisper_lang_code(language: Optional[str]) -> str:
    """Map a language name to Whisper's two-letter token code."""
    if not language:
        return "sw"
    key = str(language).strip().lower()
    return _LANG_ALIASES.get(key, key[:2])


# ------------------------------------------------------------------- collator


def make_data_collator(processor):
    """Pad labels with -100 and stack features, without needing DataCollatorSpeechSeq2SeqWithPadding.

    We build it by hand to keep the dependency surface small and to guarantee
    the padding token id is correct for the loaded checkpoint.
    """
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    def _collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        input_features = torch.tensor(
            np.asarray([f["input_features"] for f in features], dtype=np.float32)
        )
        label_arrays = [np.asarray(f["labels"], dtype=np.int64) for f in features]
        max_len = max(len(a) for a in label_arrays)
        labels = np.full((len(label_arrays), max_len), -100, dtype=np.int64)
        for i, arr in enumerate(label_arrays):
            labels[i, : len(arr)] = arr

        return {
            "input_features": input_features,
            "labels": torch.tensor(labels),
        }

    return _collate


# --------------------------------------------------------------------- args


def build_training_arguments(cfg: Config, output_dir: Path):
    from transformers import Seq2SeqTrainingArguments

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    params = set(sig.parameters)

    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": cfg.train.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.train.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
        "learning_rate": cfg.train.learning_rate,
        "weight_decay": cfg.train.weight_decay,
        "warmup_ratio": cfg.train.warmup_ratio,
        "max_grad_norm": cfg.train.max_grad_norm,
        "logging_steps": cfg.train.logging_steps,
        "save_steps": cfg.train.save_steps,
        "save_total_limit": cfg.train.save_total_limit,
        "gradient_checkpointing": cfg.train.gradient_checkpointing,
        "fp16": cfg.train.fp16,
        "bf16": cfg.train.bf16,
        "predict_with_generate": cfg.train.predict_with_generate,
        "generation_max_length": cfg.train.generation_max_length,
        "generation_num_beams": cfg.train.generation_num_beams,
        "load_best_model_at_end": True,
        "metric_for_best_model": cfg.train.metric_for_best_model,
        "greater_is_better": cfg.train.greater_is_better,
        "dataloader_num_workers": cfg.train.dataloader_num_workers,
        "label_names": ["labels"],
        "seed": cfg.data.seed,
        # MLflow is driven manually from the callback so we can guarantee
        # per-step flushing; this keeps HF from double-logging.
        "report_to": [],
        "remove_unused_columns": False,
    }

    if cfg.train.max_steps and cfg.train.max_steps > 0:
        kwargs["max_steps"] = cfg.train.max_steps
        # eval/save must be reachable within max_steps or HF errors out.
        kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = max(1, min(cfg.train.eval_steps, cfg.train.max_steps))
        kwargs["save_steps"] = max(1, min(cfg.train.save_steps, cfg.train.max_steps))
    else:
        kwargs["num_train_epochs"] = cfg.train.max_epochs
        kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = cfg.train.eval_steps

    if "overwrite_output_dir" in params:
        kwargs["overwrite_output_dir"] = True

    return Seq2SeqTrainingArguments(**kwargs)


# ------------------------------------------------------------------ helpers


def build_compute_metrics(processor, cfg: Config):
    """WER/CER based compute_metrics for the Trainer."""
    norm_kwargs = dict(
        lowercase=cfg.data.lowercase,
        strip_punctuation=cfg.data.strip_punctuation,
        collapse_repeats=cfg.data.collapse_repeats,
        max_repeat_run=cfg.data.max_repeat_run,
    )

    def _compute(pred) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Beams can produce a shape that needs squeezing.
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Restore -100 to the pad token before decoding.
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

        hyp_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
        ref_strs = processor.batch_decode(label_ids, skip_special_tokens=True)

        report = compute_metrics(
            ref_strs,
            hyp_strs,
            **norm_kwargs,
        )
        return {
            "wer": report.wer,
            "cer": report.cer,
            "mixed_wer": report.mixed_wer,
            "code_switch_rate_error": report.code_switch_rate_error,
        }

    return _compute


class MLflowMetricCallback:
    """Transformers callback that pushes logs straight into MLflow."""

    def __new__(cls, *args, **kwargs):
        from transformers import TrainerCallback

        class _CB(TrainerCallback):
            def on_log(self, training_args, state, control, logs=None, **kw):
                import mlflow

                logs = logs or {}
                step = int(state.global_step)
                metrics = {
                    k: float(v)
                    for k, v in logs.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                if metrics:
                    mlflow.log_metrics(metrics, step=step)

            def on_save(self, training_args, state, control, **kw):
                import mlflow

                ckpt = Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
                if ckpt.exists():
                    mlflow.log_metric("checkpoint_step", float(state.global_step))
                    mlflow.set_tag("last_checkpoint", ckpt.name)

        return _CB(*args, **kwargs)


def full_eval(
    model,
    processor,
    eval_ds,
    cfg: Config,
    device: Optional[str] = None,
) -> Tuple[Any, List[str], List[str]]:
    """Generate predictions over the eval set and score them properly.

    The Trainer's compute_metrics gives us a WER number, but final reporting
    goes through this path so we also get per-example rows and the
    code-switch diagnostics.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    refs: List[str] = []
    hyps: List[str] = []

    with torch.no_grad():
        for row in eval_ds:
            feats = torch.tensor(
                np.asarray([row["input_features"]], dtype=np.float32), device=device
            )
            out = model.generate(
                feats,
                language=f"<|{_whisper_lang_code(cfg.model.language)}|>",
                task=cfg.model.task,
                max_new_tokens=cfg.train.generation_max_length,
                num_beams=cfg.train.generation_num_beams,
            )
            hyp = processor.tokenizer.decode(out[0], skip_special_tokens=True)
            hyps.append(hyp)
            refs.append(row.get("raw_text") or "")

    report = compute_metrics(
        refs,
        hyps,
        lowercase=cfg.data.lowercase,
        strip_punctuation=cfg.data.strip_punctuation,
        collapse_repeats=cfg.data.collapse_repeats,
        max_repeat_run=cfg.data.max_repeat_run,
    )
    return report, refs, hyps


def save_checkpoint_bundle(model, processor, cfg: Config, out_dir: Path) -> Path:
    """Save a mergeable, installable model directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))

    # Keep the resolved config next to the weights for reproducibility.
    (out_dir / "sheng_training_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2), encoding="utf-8"
    )
    return out_dir
