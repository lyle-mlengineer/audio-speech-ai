"""Training configuration: one dataclass -> CLI flags for the trainer + a YAML record.

The trainer itself (``sheng_qwentts.train``) is argparse-only so it can be run by
hand on any machine; this module is the single source of truth for the knobs and
the per-GPU profiles.

Profile invariants (see README):

* effective batch (``batch_size * grad_accum``) is **4** everywhere, so ``steps``
  means the same thing on every GPU. 4 is small on purpose: the dataset has 38
  training clips, so a bigger effective batch leaves only a handful of steps per
  epoch;
* T4 (Turing, sm75) has no usable bf16 matmul -> ``dtype: float16`` plus loss
  scaling, and it is only allowed with ``--allow-small-gpu``;
* the upstream learning rates 2e-5 / 1e-5 / 2e-6 have all been reported to either
  produce noise or nothing at all for this model family; default is 2e-6 with a
  warmup, see README "What goes wrong".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import DEFAULT_RUN, DEFAULT_STEPS, Paths

EFFECTIVE_BATCH = 4


@dataclass(frozen=True)
class Profile:
    name: str
    batch_size: int
    grad_accum: int
    dtype: str
    attn: str
    note: str


PROFILES: dict[str, Profile] = {
    "t4": Profile("t4", 1, 4, "float16", "sdpa", "Colab T4 16 GB - needs --allow-small-gpu"),
    "l4": Profile("l4", 2, 2, "bfloat16", "sdpa", "Colab L4 24 GB (recommended)"),
    "a100": Profile("a100", 4, 1, "bfloat16", "flash_attention_2", "Colab A100 40/80 GB (fastest)"),
    "cpu": Profile("cpu", 1, 4, "float32", "eager", "no GPU - smoke test only, cannot train"),
}

SMALL_GPU = {"t4", "cpu"}


@dataclass
class TrainConfig:
    run: str = DEFAULT_RUN
    profile: str = "l4"
    steps: int = DEFAULT_STEPS
    batch_size: int = 2
    grad_accum: int = 2
    learning_rate: float = 2e-06
    weight_decay: float = 0.01
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    log_interval: int = 1
    save_interval: int = 40
    keep_last: int = 3
    sub_talker_loss_weight: float = 0.3
    speaker_name: str = "abel"
    dtype: str = "bfloat16"
    attn: str = "sdpa"
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
        if not 0 < self.learning_rate < 1e-3:
            raise ValueError(f"learning_rate {self.learning_rate} outside (0, 1e-3)")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"dtype must be bfloat16/float16/float32, got {self.dtype!r}")
        if self.attn not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError("attn must be sdpa / flash_attention_2 / eager")
        if self.profile == "t4" and self.dtype != "float16":
            raise ValueError("T4 has no usable bf16 matmul - use dtype: float16")
        if self.profile in {"l4", "a100"} and self.dtype == "float32":
            raise ValueError("float32 training does not fit on a 24 GB card - use bfloat16")
        if self.save_interval < 5:
            raise ValueError("save_interval < 5 makes the run slower than the checkpoints are worth")
        if self.keep_last < 1:
            raise ValueError("keep_last must be >= 1 (you always want `latest`)")
        if not 0 <= self.sub_talker_loss_weight <= 1:
            raise ValueError("sub_talker_loss_weight must be within [0, 1]")
        return self

    @classmethod
    def from_profile(cls, profile: str = "l4", **overrides: Any) -> "TrainConfig":
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
        p = PROFILES[profile]
        base = cls(
            profile=profile,
            batch_size=p.batch_size,
            grad_accum=p.grad_accum,
            dtype=p.dtype,
            attn=p.attn,
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(base, key, value)
        return base.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # --------------------------------------------------------------- args
    def trainer_args(self, paths: Paths) -> list[str]:
        """argv for ``python -m sheng_qwentts.train`` (paths resolved)."""
        return [
            "--model-dir", str(paths.model_dir),
            "--train-jsonl", str(paths.train_codes_jsonl),
            "--output-dir", str(paths.checkpoint_dir(self.run)),
            "--run", self.run,
            "--speaker-name", self.speaker_name,
            "--steps", str(self.steps),
            "--batch-size", str(self.batch_size),
            "--grad-accum", str(self.grad_accum),
            "--lr", format_float(self.learning_rate),
            "--weight-decay", format_float(self.weight_decay),
            "--warmup-steps", str(self.warmup_steps),
            "--max-grad-norm", format_float(self.max_grad_norm),
            "--log-interval", str(self.log_interval),
            "--save-interval", str(self.save_interval),
            "--keep-last", str(self.keep_last),
            "--sub-talker-loss-weight", format_float(self.sub_talker_loss_weight),
            "--dtype", self.dtype,
            "--attn-implementation", self.attn,
            "--seed", str(self.seed),
            "--num-workers", str(self.num_workers),
            "--tensorboard", str(paths.log_dir(self.run) / "tb") if self.tensorboard else "",
        ]

    # --------------------------------------------------------------- yaml
    def render_yaml(self, paths: Paths) -> str:
        """Deterministic record of exactly what will run (resolved paths inside)."""
        d = self.to_dict()

        def p(obj) -> str:
            return Path(obj).as_posix()

        rows: list[tuple[str, str]] = [
            ("run", self.run),
            ("profile", self.profile),
            ("# effective batch = batch_size * grad_accum (fixed at 4)", ""),
            ("batch_size", str(d["batch_size"])),
            ("grad_accum", str(d["grad_accum"])),
            ("steps", str(d["steps"])),
            ("learning_rate", format_float(d["learning_rate"])),
            ("weight_decay", format_float(d["weight_decay"])),
            ("warmup_steps", str(d["warmup_steps"])),
            ("max_grad_norm", format_float(d["max_grad_norm"])),
            ("log_interval", str(d["log_interval"])),
            ("save_interval", str(d["save_interval"])),
            ("keep_last", str(d["keep_last"])),
            ("sub_talker_loss_weight", format_float(d["sub_talker_loss_weight"])),
            ("speaker_name", self.speaker_name),
            ("dtype", self.dtype),
            ("attn", self.attn),
            ("seed", str(d["seed"])),
            ("tensorboard", "true" if self.tensorboard else "false"),
            ("", ""),
            ("# resolved paths (artifacts root: " + p(paths.artifacts) + ")", ""),
            ("model_dir", p(paths.model_dir)),
            ("tokenizer_dir", p(paths.tokenizer_dir)),
            ("train_jsonl", p(paths.train_codes_jsonl)),
            ("save_dir", p(paths.checkpoint_dir(self.run))),
        ]
        lines = [f"# rendered by sheng-qwentts from profile {self.profile} - do not hand-edit"]
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
    """2e-06 -> 0.000002 (PyYAML<->float round-trip safe, no scientific notation)."""
    text = f"{float(value):.12f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def allowed_profiles(*, allow_small_gpu: bool = False) -> list[str]:
    return [name for name in PROFILES if allow_small_gpu or name not in SMALL_GPU]


def steps_for_hours(hours: float, steps_per_hour: float) -> int:
    """Rough stop-time hint for ``--max-hours`` (used in log lines only)."""
    return max(1, int(math.floor(hours * steps_per_hour)))
