"""Training configuration: one dataclass -> CLI flags for the trainer + a YAML record.

The trainer itself (``sheng_chatterbox.train_t3``) is argparse-only so it can be
run by hand on any machine; this module is the single source of truth for the
knobs and the per-GPU profiles.

Profile invariants (see README):

* effective batch (``batch_size * grad_accum``) is **8** everywhere, so ``steps``
  means the same thing on every GPU;
* T4 (Turing) has no native bf16 -> ``dtype: fp16`` and it is only allowed with
  ``--allow-small-gpu``;
* ``save_interval`` is 100 and ``keep_last`` 3, so a disconnect costs <= 100 steps
  and the Drive footprint stays ~4 GB (fp16 checkpoints, ~1 GB each).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import DEFAULT_RUN, DEFAULT_STEPS, LANGUAGE_ID, Paths

EFFECTIVE_BATCH = 8


@dataclass(frozen=True)
class Profile:
    name: str
    batch_size: int
    grad_accum: int
    dtype: str
    note: str


PROFILES: dict[str, Profile] = {
    "l4": Profile("l4", 4, 2, "bf16", "Colab L4 24 GB (recommended)"),
    "a100": Profile("a100", 8, 1, "bf16", "Colab A100 40/80 GB (fastest)"),
    "t4": Profile("t4", 2, 4, "fp16", "Colab T4 16 GB - needs --allow-small-gpu"),
    "cpu": Profile("cpu", 1, 8, "fp32", "no GPU - dry-run / smoke test only"),
}

SMALL_GPU = {"t4", "cpu"}


@dataclass
class TrainConfig:
    run: str = DEFAULT_RUN
    profile: str = "l4"
    steps: int = DEFAULT_STEPS
    batch_size: int = 4
    grad_accum: int = 2
    learning_rate: float = 5e-05
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    log_interval: int = 10
    valid_interval: int = 100
    save_interval: int = 100
    keep_last: int = 3
    max_text_len: int = 512
    max_speech_len: int = 1024
    cond_seconds: float = 6.0  # mtl_tts.ENC_COND_LEN == 6 * S3_SR
    emotion_adv: float = 0.5
    model_variant: str = "mtl"  # "mtl" (Swahili supported) or "english"
    language_id: str = LANGUAGE_ID
    dtype: str = "bf16"
    seed: int = 1337
    num_workers: int = 0
    tensorboard: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- helpers
    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def validate(self) -> "TrainConfig":
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("batch_size and grad_accum must be >= 1")
        if self.effective_batch != EFFECTIVE_BATCH:
            raise ValueError(
                f"batch_size*grad_accum must be {EFFECTIVE_BATCH} (got {self.effective_batch}); "
                "keeping it fixed makes 'steps' comparable across GPUs"
            )
        if not 0 < self.learning_rate < 1e-2:
            raise ValueError(f"learning_rate {self.learning_rate} outside (0, 1e-2)")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"dtype must be bf16/fp16/fp32, got {self.dtype!r}")
        if self.model_variant not in {"mtl", "english"}:
            raise ValueError("model_variant must be 'mtl' or 'english'")
        if not 1.0 <= self.cond_seconds <= 10.0:
            raise ValueError("cond_seconds must be within 1-10 s (prompt conditioning window)")
        if self.profile == "t4" and self.dtype != "fp16":
            raise ValueError("T4 has no native bf16 - use dtype: fp16")
        if self.save_interval < 10:
            raise ValueError("save_interval < 10 makes the run slower than the checkpoints are worth")
        if self.keep_last < 1:
            raise ValueError("keep_last must be >= 1 (you always want `latest`)")
        return self

    @classmethod
    def from_profile(cls, profile: str = "l4", **overrides: Any) -> "TrainConfig":
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
        p = PROFILES[profile]
        base = cls(profile=profile, batch_size=p.batch_size, grad_accum=p.grad_accum, dtype=p.dtype)
        for key, value in overrides.items():
            if value is not None:
                setattr(base, key, value)
        return base.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # --------------------------------------------------------------- args
    def trainer_args(self, paths: Paths) -> list[str]:
        """argv for ``python -m sheng_chatterbox.train_t3`` (paths resolved)."""
        args = [
            "--train-manifest", str(paths.train_manifest),
            "--val-manifest", str(paths.val_manifest) if paths.val_manifest.exists() else "",
            "--pretrained-dir", str(paths.pretrained_dir),
            "--base-t3", paths.base_t3_path.name,
            "--save-dir", str(paths.checkpoint_dir(self.run)),
            "--cache-dir", str(paths.feature_dir(self.run)),
            "--log-dir", str(paths.log_dir(self.run)),
            "--run", self.run,
            "--model-variant", self.model_variant,
            "--language-id", self.language_id,
            "--steps", str(self.steps),
            "--batch-size", str(self.batch_size),
            "--grad-accum", str(self.grad_accum),
            "--lr", format_float(self.learning_rate),
            "--weight-decay", format_float(self.weight_decay),
            "--warmup-steps", str(self.warmup_steps),
            "--max-grad-norm", format_float(self.max_grad_norm),
            "--log-interval", str(self.log_interval),
            "--valid-interval", str(self.valid_interval),
            "--save-interval", str(self.save_interval),
            "--keep-last", str(self.keep_last),
            "--max-text-len", str(self.max_text_len),
            "--max-speech-len", str(self.max_speech_len),
            "--cond-seconds", format_float(self.cond_seconds),
            "--emotion-adv", format_float(self.emotion_adv),
            "--dtype", self.dtype,
            "--seed", str(self.seed),
            "--num-workers", str(self.num_workers),
        ]
        if self.tensorboard:
            args += ["--tensorboard", str(paths.log_dir(self.run) / "tb")]
        return args

    # --------------------------------------------------------------- yaml
    def render_yaml(self, paths: Paths) -> str:
        """Deterministic record of exactly what will run (resolved paths inside)."""
        d = self.to_dict()
        def p(obj) -> str:
            return Path(obj).as_posix()

        root = p(paths.artifacts)
        val = p(paths.val_manifest) if paths.val_manifest.exists() else '""'
        rows: list[tuple[str, str]] = [
            ("run", self.run),
            ("profile", self.profile),
            ("# effective batch = batch_size * grad_accum (fixed at 8)", ""),
            ("batch_size", str(d["batch_size"])),
            ("grad_accum", str(d["grad_accum"])),
            ("steps", str(d["steps"])),
            ("learning_rate", format_float(d["learning_rate"])),
            ("weight_decay", format_float(d["weight_decay"])),
            ("warmup_steps", str(d["warmup_steps"])),
            ("max_grad_norm", format_float(d["max_grad_norm"])),
            ("log_interval", str(d["log_interval"])),
            ("valid_interval", str(d["valid_interval"])),
            ("save_interval", str(d["save_interval"])),
            ("keep_last", str(d["keep_last"])),
            ("# S3 speech tokens are 25/s; 1024 covers ~40 s incl. prompt", ""),
            ("max_text_len", str(d["max_text_len"])),
            ("max_speech_len", str(d["max_speech_len"])),
            ("cond_seconds", format_float(d["cond_seconds"])),
            ("emotion_adv", format_float(d["emotion_adv"])),
            ("model_variant", self.model_variant),
            ("language_id", self.language_id),
            ("dtype", self.dtype),
            ("seed", str(d["seed"])),
            ("tensorboard", "true" if self.tensorboard else "false"),
            ("", ""),
            (f"# resolved paths (artifacts root: {root})", ""),
            ("pretrained_dir", p(paths.pretrained_dir)),
            ("base_t3", paths.base_t3_path.name),
            ("train_manifest", p(paths.train_manifest)),
            ("val_manifest", val),
            ("save_dir", p(paths.checkpoint_dir(self.run))),
            ("cache_dir", p(paths.feature_dir(self.run))),
        ]
        lines = [f"# rendered by sheng-chatterbox from profile {self.profile} - do not hand-edit"]
        for key, value in rows:
            if not key:
                lines.append("")
            elif key.startswith("#"):
                lines.append(key)
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines) + "\n"

    def write_yaml(self, paths: Paths) -> Path:
        path = paths.config_path(self.run)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render_yaml(paths), encoding="utf-8", newline="\n")
        return path


def format_float(value: float) -> str:
    """5e-05 -> 0.00005 (PyYAML<->float round-trip safe, no scientific notation)."""
    text = f"{float(value):.10f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def allowed_profiles(*, allow_small_gpu: bool = False) -> list[str]:
    return [name for name in PROFILES if allow_small_gpu or name not in SMALL_GPU]


def steps_for_hours(hours: float, steps_per_hour: float) -> int:
    """Rough stop-time hint for ``--max-hours`` (used in log lines only)."""
    return max(1, int(math.floor(hours * steps_per_hour)))
