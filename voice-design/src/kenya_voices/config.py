"""Training-config generation: GPU profiles + the YAML VoxCPM2 actually reads.

Verified against upstream ``conf/voxcpm_v2/voxcpm_finetune_lora.yaml`` and
``scripts/train_voxcpm_finetune.py``. Nothing here is voice-design specific: the
caption lives inside each row's ``text``, so the trainer tokenises description +
transcript exactly the way inference does.

* ``sample_rate: 16000`` is asserted against ``audio_vae.sample_rate`` - VoxCPM2's
  AudioVAE encoder runs at 16 kHz (hop 640 -> 25 latent frames/s) and the decoder
  outputs 48 kHz. Setting 48000 here fails the assert; it belongs in
  ``out_sample_rate`` (TensorBoard logging only).
* ``num_iters`` is the loop bound; ``max_steps`` only sizes the cosine schedule.
  They are kept equal so "steps" means one thing.
* Checkpoints are ``<save_path>/step_%07d`` plus a ``latest`` copy that the
  trainer resumes from automatically.
* The trainer autocasts bfloat16, so the profile is chosen from real VRAM, and
  T4-class cards get batch 1 / 4k tokens so they at least run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import DEFAULT_RUN, OUT_SAMPLE_RATE, TARGET_SAMPLE_RATE, Paths

# Per-GPU knobs. VRAM guidance from the official docs: ~20 GB for LoRA.
PROFILES: dict[str, dict[str, Any]] = {
    "l4": {
        "batch_size": 2,
        "grad_accum_steps": 8,
        "max_batch_tokens": 8192,
        "num_workers": 8,
        "preprocessing_num_workers": 8,
        "note": "L4 / A10 / 24 GB - the official LoRA shape (effective batch 16)",
    },
    "a100": {
        "batch_size": 4,
        "grad_accum_steps": 4,
        "max_batch_tokens": 16384,
        "num_workers": 8,
        "preprocessing_num_workers": 8,
        "note": "A100 40/80 GB - bigger micro-batch, same effective batch 16",
    },
    "t4": {
        "batch_size": 1,
        "grad_accum_steps": 16,
        "max_batch_tokens": 4096,
        "num_workers": 4,
        "preprocessing_num_workers": 4,
        "note": "T4 / 16 GB - emulated bf16, slow; use --allow-small-gpu",
    },
}

# ~2 epochs over ~3.5k caption rows at effective batch 16 (10 h of audio, 120
# speakers). Voice design needs many speakers per caption, not many epochs, so
# watch val loss and prefer the earliest checkpoint that still designs new voices.
DEFAULT_STEPS = 1200
LORA_R = 32


@dataclass
class TrainConfig:
    run: str = DEFAULT_RUN
    profile: str = "l4"
    steps: int = DEFAULT_STEPS
    learning_rate: float = 1e-4
    lora_r: int = LORA_R
    lora_alpha: int = LORA_R
    lora_dropout: float = 0.0
    enable_lm: bool = True
    enable_dit: bool = True
    enable_proj: bool = False
    save_interval: int = 100
    valid_interval: int = 100
    log_interval: int = 10
    warmup_steps: int = 50
    overrides: dict[str, Any] = field(default_factory=dict)

    def resolved(self) -> dict[str, Any]:
        prof = dict(PROFILES[self.profile])
        prof.pop("note", None)
        cfg: dict[str, Any] = {
            "pretrained_path": None,  # filled in build_payload
            "train_manifest": None,
            "val_manifest": None,
            "sample_rate": TARGET_SAMPLE_RATE,
            "out_sample_rate": OUT_SAMPLE_RATE,
            **prof,
            "num_iters": self.steps,
            "max_steps": self.steps,
            "log_interval": self.log_interval,
            "valid_interval": self.valid_interval,
            "save_interval": self.save_interval,
            "learning_rate": self.learning_rate,
            "weight_decay": 0.01,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": 1.0,
            "save_path": None,
            "tensorboard": None,
            "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
            "lora": {
                "enable_lm": self.enable_lm,
                "enable_dit": self.enable_dit,
                "enable_proj": self.enable_proj,
                "r": self.lora_r,
                "alpha": self.lora_alpha if self.lora_alpha is not None else self.lora_r,
                "dropout": self.lora_dropout,
            },
            # Keep the base model path (not the Hub id) so checkpoints are
            # self-describing on Drive; use `export` to publish with the Hub id.
            "distribute": False,
        }
        cfg.update(self.overrides)
        return cfg

    def build_payload(self, paths: Paths) -> dict[str, Any]:
        cfg = self.resolved()
        cfg["pretrained_path"] = str(paths.model_dir)
        cfg["train_manifest"] = str(paths.train_manifest)
        # Upstream treats an empty val_manifest as "no validation split"; a
        # missing val.jsonl means the dataset was too small to hold clips out.
        cfg["val_manifest"] = str(paths.val_manifest) if paths.val_manifest.exists() else ""
        cfg["save_path"] = str(paths.checkpoint_dir(self.run))
        cfg["tensorboard"] = str(paths.tensorboard_dir(self.run))
        return cfg


def render_yaml(obj: Any, indent: int = 0) -> str:
    """Small deterministic YAML emitter (no PyYAML needed at train time)."""
    pad = " " * indent
    lines: list[str] = []
    for key, value in obj.items():
        key_str = str(key)
        if isinstance(value, dict):
            lines.append(f"{pad}{key_str}:")
            lines.append(render_yaml(value, indent + 2))
        elif isinstance(value, bool):
            lines.append(f"{pad}{key_str}: {'true' if value else 'false'}")
        elif isinstance(value, float):
            # YAML 1.1 has no exponents without a dot: 5e-05 would load as the
            # *string* "5e-05" instead of a float. Always emit a plain decimal.
            text = repr(value)
            if "e" in text or "E" in text:
                text = f"{value:.12f}".rstrip("0").rstrip(".")
            lines.append(f"{pad}{key_str}: {text}")
        elif isinstance(value, int):
            lines.append(f"{pad}{key_str}: {value}")
        elif value is None:
            lines.append(f"{pad}{key_str}: null")
        else:
            # json.dumps gives a valid YAML double-quoted scalar (escapes \\ and ")
            lines.append(f"{pad}{key_str}: {json.dumps(str(value), ensure_ascii=False)}")
    return "\n".join(lines)


def write_config(paths: Paths, cfg: TrainConfig) -> Path:
    paths.ensure()
    out = paths.run_config(cfg.run)
    payload = cfg.build_payload(paths)
    header = (
        f"# generated by kenya_voices (profile={cfg.profile}, steps={cfg.steps}, "
        f"lora_r={cfg.lora_r}) - edit configs/ or CLI flags, not this file\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + render_yaml(payload) + "\n", encoding="utf-8", newline="\n")
    return out


def profile_summary(cfg: TrainConfig) -> str:
    prof = PROFILES[cfg.profile]
    return (
        f"profile={cfg.profile} ({prof['note']}) | steps={cfg.steps} | "
        f"batch={prof['batch_size']}x{prof['grad_accum_steps']} | "
        f"max_batch_tokens={prof['max_batch_tokens']} | lora r={cfg.lora_r}/{cfg.lora_alpha} | lr={cfg.learning_rate}"
    )
