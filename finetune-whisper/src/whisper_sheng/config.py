"""Typed configuration for Whisper fine-tuning.

Everything that varies between runs lives here. The model is deliberately
configurable: `model.name` accepts any Whisper checkpoint on the Hub
(openai/whisper-small, openai/whisper-medium, ...) so you can swap sizes
without touching code.

Config is loaded from YAML, optionally overridden by CLI flags, then
validated so bad combinations fail fast instead of 20 minutes into training.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Which Whisper to load and how to condition its decoder."""

    # Any HF Whisper checkpoint. tiny/base/small/medium/large-v3 all work.
    name: str = "openai/whisper-medium"

    # Language token to force. Whisper has no "sheng" language token, and
    # Sheng is Swahili-dominant with heavy English mixing, so we pin the
    # language to Swahili and force `transcribe`. The model still emits
    # English words naturally because the *text* is not constrained.
    language: str = "swahili"
    task: str = "transcribe"

    # Drop the transcription timestamps. Sheng is conversational and our
    # labels have no timestamps, so asking for them adds loss we can't supervise.
    return_timestamps: bool = False

    # 8-bit AdamW keeps the optimizer state small enough for whisper-medium
    # on a 15GB T4 (medium has ~769M params; fp32 AdamW state would be ~6GB).
    use_8bit_adam: bool = True


@dataclass
class DataConfig:
    """Dataset and preprocessing options."""

    dataset_name: str = "lyle-mlengineer/kenyan-celebs"
    dataset_config: Optional[str] = None
    # The dataset ships a single `train` split, so we carve out eval ourselves.
    train_split: str = "train"
    eval_split: Optional[str] = None  # None -> split from train_split
    audio_column: str = "audio"
    text_column: str = "text"

    # Held-out fraction. With only 42 rows this yields ~6 eval examples,
    # which is a very noisy WER estimate. We keep it because *some*
    # held-out signal is better than none, and we never select on it blindly.
    eval_fraction: float = 0.15
    seed: int = 42

    # Whisper's hard input limit is 30s; the dataset is already 30s clips.
    max_input_seconds: float = 30.0
    sampling_rate: int = 16_000

    # Text normalisation is critical for Sheng: it is colloquial and
    # inconsistently spelled, so punctuation/case noise would otherwise
    # dominate the WER number.
    lowercase: bool = True
    strip_punctuation: bool = True
    # Collapse repeated characters ("nomaaa" -> "noma"). Sheng uses
    # elongation expressively and Whisper cannot learn 4-way length
    # distinctions from 42 examples.
    collapse_repeats: bool = True
    max_repeat_run: int = 2

    # Speed perturbation. Small dataset + big model = the model will
    # memorise; perturbing tempo is the cheapest regulariser that stays
    # acoustically plausible for speech.
    speed_perturb: bool = True
    speed_perturb_prob: float = 0.3
    speed_factors: List[float] = field(default_factory=lambda: [0.9, 0.95, 1.05, 1.1])

    # SpecAugment on the log-mel features.
    spec_augment: bool = True
    freq_mask_param: int = 24
    time_mask_param: int = 40
    num_freq_masks: int = 2
    num_time_masks: int = 2


@dataclass
class TrainConfig:
    """Optimisation settings tuned for a single T4."""

    output_dir: str = "outputs/whisper-sheng"
    max_epochs: int = 12
    max_steps: int = -1  # >0 overrides max_epochs

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    # Effective batch ~8 on a T4. Small batches + million-param model is
    # where gradient noise becomes useful regularisation.
    gradient_accumulation_steps: int = 2

    # 1e-5 is the standard full-fine-tune LR for Whisper. Going higher
    # destabilises the pretrained acoustic encoder on tiny data.
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    gradient_checkpointing: bool = True
    fp16: bool = True  # T4 has no bf16, so fp16 it is
    bf16: bool = False

    logging_steps: int = 5
    eval_steps: int = 20
    save_steps: int = 20
    save_total_limit: int = 2

    # With ~6 eval rows, "best" is noisy. We still early-stop, but the
    # load_best_model_at_end metric is a deliberate choice, not an accident.
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    early_stopping_patience: int = 3

    dataloader_num_workers: int = 2
    predict_with_generate: bool = True
    generation_max_length: int = 225
    generation_num_beams: int = 1  # beams are slow on T4 and rarely help ASR WER here

    resume_from_checkpoint: Optional[str] = None
    # Train on a tiny slice first to prove the plumbing works before
    # burning GPU time. Set from the CLI via --smoke-test.
    smoke_test_rows: Optional[int] = None


@dataclass
class MLflowConfig:
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "whisper-sheng"
    run_name: Optional[str] = None
    log_model_artifacts: bool = True
    # Colab gives ~112GB disk; keeping a full copy of the weights in the
    # MLflow artifact store doubles usage, so this is opt-out.
    log_checkpoint_artifact: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class HubConfig:
    push_to_hub: bool = True
    repo_id: str = "lyle-mlengineer/whisper-medium-sheng"
    private: bool = False
    commit_message: str = "Fine-tune Whisper for Sheng ASR"
    # Also upload the eval report as a dataset card / artifact on the model repo.
    upload_eval_report: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    # ---------------------------------------------------------------- helpers

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        cfg = cls()
        known_sections = {f.name for f in fields(cls)}
        unknown = set(raw) - known_sections
        if unknown:
            raise ValueError(
                f"Unknown config section(s): {sorted(unknown)}. "
                f"Expected any of {sorted(known_sections)}."
            )
        for section_name, section_values in raw.items():
            section_obj = getattr(cfg, section_name)
            if not isinstance(section_values, dict):
                raise ValueError(f"Config section '{section_name}' must be a mapping.")
            valid = {f.name for f in fields(section_obj)}
            bad = set(section_values) - valid
            if bad:
                raise ValueError(
                    f"Unknown key(s) in '{section_name}': {sorted(bad)}. "
                    f"Valid keys: {sorted(valid)}."
                )
            for k, v in section_values.items():
                setattr(section_obj, k, v)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten to `section.key` -> scalar for MLflow params."""
        flat: Dict[str, Any] = {}
        for section_name, section_obj in asdict(self).items():
            for k, v in section_obj.items():
                if isinstance(v, (list, dict)):
                    v = json.dumps(v)
                flat[f"{section_name}.{k}"] = v
        return flat

    def apply_overrides(self, overrides: Dict[str, Any]) -> "Config":
        """Apply CLI overrides in `section.key=value` form."""
        for dotted, value in overrides.items():
            if "." not in dotted:
                raise ValueError(f"Override '{dotted}' must look like section.key=value")
            section_name, key = dotted.split(".", 1)
            if not hasattr(self, section_name):
                raise ValueError(f"Unknown section in override: {section_name}")
            section_obj = getattr(self, section_name)
            if not hasattr(section_obj, key):
                raise ValueError(f"Unknown key in override: {section_name}.{key}")
            current = getattr(section_obj, key)
            setattr(section_obj, key, _coerce(value, current))
        return self

    def validate(self) -> None:
        errs: List[str] = []
        if self.model.language is None:
            errs.append("model.language must be set (use 'swahili' for Sheng).")
        if not 0.0 < self.data.eval_fraction < 1.0:
            errs.append("data.eval_fraction must be between 0 and 1.")
        if self.train.fp16 and self.train.bf16:
            errs.append("train.fp16 and train.bf16 are mutually exclusive.")
        if self.train.per_device_train_batch_size < 1:
            errs.append("train.per_device_train_batch_size must be >= 1.")
        try:
            import torch

            if self.train.bf16 and not torch.cuda.is_bf16_supported():
                errs.append("train.bf16 requested but this GPU lacks bf16 support (T4 cannot).")
        except Exception:
            pass
        if self.hub.push_to_hub and not self.hub.repo_id:
            errs.append("hub.push_to_hub is True but hub.repo_id is empty.")
        if errs:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errs))


def _coerce(value: Any, like: Any) -> Any:
    """Coerce a CLI string into the type of the existing field value."""
    if isinstance(value, str) and like is not None:
        if isinstance(like, bool):
            if value.lower() in {"1", "true", "yes", "y"}:
                return True
            if value.lower() in {"0", "false", "no", "n"}:
                return False
            if value.lower() in {"none", "null"}:
                return None
            raise ValueError(f"Expected a boolean, got {value!r}")
        if isinstance(like, int) and not isinstance(like, bool):
            return int(value)
        if isinstance(like, float):
            return float(value)
    if value == "none" and value is not None:
        return None
    return value


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
