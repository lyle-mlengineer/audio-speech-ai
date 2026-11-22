"""Training profiles.

The numbers are *samples per step*, not F5 frames. The dataset is 42 clips of
exactly 30 s, so one sample is 2812 mel frames (30 * 24000 / 256). F5's own
VRAM formula is ``38400 * (VRAM_GB - 5) / 75`` frames, i.e. ~9728 frames on a
24 GB card: two 30 s samples per step fit with room to spare.

Effective batch is 4 samples everywhere, so ``epochs`` means roughly the same
amount of training on every GPU.

The default runtime is Colab's free **T4** GPU (``DEFAULT_PROFILE``). The t4
profile is sized for its 16 GB and Turing (sm75) compute: one 30 s sample per
forward pass, 4-step gradient accumulation and fp16, because T4 has no bf16. It
is slow, not fragile - no opt-in flag is needed anymore.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    batch_size: int          # samples per forward pass
    grad_accum: int          # micro-batches per optimizer update
    epochs: int
    learning_rate: float
    warmup_updates: int
    save_per_updates: int
    last_per_updates: int
    keep_last: int
    mixed_precision: str     # "bf16" | "fp16"
    max_samples: int         # cap inside a batch (F5 Trainer argument)
    min_vram_gb: float
    note: str

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def as_dict(self) -> dict:
        d = asdict(self)
        d["effective_batch"] = self.effective_batch
        return d


#: Profile used when nothing better is detected / requested: the free Colab T4.
DEFAULT_PROFILE = "t4"

PROFILES: dict[str, Profile] = {
    "t4": Profile(
        name="t4",
        batch_size=1,
        grad_accum=4,
        epochs=200,
        learning_rate=1e-5,
        warmup_updates=100,
        save_per_updates=200,
        last_per_updates=50,
        keep_last=3,
        mixed_precision="fp16",
        max_samples=1,
        min_vram_gb=14.0,
        note="Colab T4 16 GB. The default runtime; slow, so verify wiring with --smoke first.",
    ),
    "l4": Profile(
        name="l4",
        batch_size=2,
        grad_accum=2,
        epochs=150,
        learning_rate=1e-5,
        warmup_updates=100,
        save_per_updates=250,
        last_per_updates=100,
        keep_last=3,
        mixed_precision="bf16",
        max_samples=2,
        min_vram_gb=20.0,
        note="Colab L4 24 GB. ~3x faster than T4; full fine-tune, no LoRA.",
    ),
    "a100": Profile(
        name="a100",
        batch_size=4,
        grad_accum=1,
        epochs=150,
        learning_rate=1e-5,
        warmup_updates=100,
        save_per_updates=250,
        last_per_updates=100,
        keep_last=3,
        mixed_precision="bf16",
        max_samples=4,
        min_vram_gb=35.0,
        note="Colab A100 40/80 GB. Same effective batch, fewest steps per update.",
    ),
}

assert DEFAULT_PROFILE in PROFILES, "DEFAULT_PROFILE must name a known profile"


def auto_profile(gpu_name: str | None, vram_gb: float | None) -> str | None:
    """Best profile for the detected GPU, or None when nothing fits.

    A T4 resolves to the default (t4) profile, so the free Colab runtime trains
    out of the box.
    """
    name = (gpu_name or "").upper()
    vram = float(vram_gb or 0.0)
    if "A100" in name or "H100" in name or "L40" in name:
        return "a100"
    if "L4" in name:
        return "l4"
    if "T4" in name or "V100" in name:
        return DEFAULT_PROFILE if vram >= PROFILES[DEFAULT_PROFILE].min_vram_gb else None
    if vram >= PROFILES["a100"].min_vram_gb:
        return "a100"
    if vram >= PROFILES["l4"].min_vram_gb:
        return "l4"
    if vram >= PROFILES[DEFAULT_PROFILE].min_vram_gb:
        return DEFAULT_PROFILE
    return None


def resolve_profile(
    requested: str | None,
    gpu_name: str | None,
    vram_gb: float | None,
    allow_small_gpu: bool = True,  # deprecated: the t4 profile is no longer gated
) -> Profile:
    """Explicit profile wins; otherwise auto-detect and enforce the VRAM floor."""
    del allow_small_gpu  # kept so old invocations/scripts still work
    if requested:
        if requested not in PROFILES:
            raise ValueError(f"unknown profile {requested!r}; known: {sorted(PROFILES)}")
        profile = PROFILES[requested]
    else:
        detected = auto_profile(gpu_name, vram_gb)
        if detected is None:
            raise RuntimeError(
                f"no profile for {gpu_name or 'unknown GPU'} with {vram_gb or 0:.0f} GB VRAM; "
                f"F5-TTS fine-tuning wants >= {PROFILES[DEFAULT_PROFILE].min_vram_gb:.0f} GB "
                f"(the {DEFAULT_PROFILE} profile). Pass --profile {{t4,l4,a100}} to try another shape."
            )
        profile = PROFILES[detected]

    if vram_gb is not None and vram_gb < profile.min_vram_gb:
        raise RuntimeError(
            f"profile {profile.name!r} wants >= {profile.min_vram_gb:.0f} GB VRAM, "
            f"found {vram_gb:.0f} GB"
        )
    return profile


def updates_for(n_samples: int, batch_size: int, grad_accum: int) -> int:
    """Optimizer updates in one epoch for an explicit batch geometry."""
    if n_samples <= 0:
        return 0
    import math

    batches = math.ceil(n_samples / max(1, batch_size))
    return max(1, math.ceil(batches / max(1, grad_accum)))


def updates_per_epoch(n_samples: int, profile: Profile) -> int:
    """Optimizer updates in one epoch (F5 batches ``batch_size`` samples)."""
    return updates_for(n_samples, profile.batch_size, profile.grad_accum)


def total_updates(n_samples: int, profile: Profile, epochs: int | None = None) -> int:
    return updates_per_epoch(n_samples, profile) * (epochs or profile.epochs)


# --------------------------------------------------------------------- configs
def render_config(profile: Profile, run: str, paths, **overrides) -> str:
    """Human-readable YAML mirror of the settings the trainer will receive.

    Written to ``<artifacts>/configs/<run>.yaml``. Nothing reads it back - the
    trainer gets explicit flags - but it makes a run reproducible after a
    disconnect.
    """
    values = profile.as_dict()
    values.pop("note", None)
    values["profile"] = values.pop("name", profile.name)
    values.update({k: v for k, v in overrides.items() if v is not None})
    values["run"] = run
    values["pretrained"] = str(paths.sauti_ema_ckpt)
    values["dataset_dir"] = str(paths.f5_dataset_dir)
    values["vocab"] = str(paths.f5_vocab)
    values["checkpoint_dir"] = str(paths.checkpoint_dir(run))

    lines = [
        f"# rendered by sauti-sheng from profile {profile.name!r}; flags win",
        "# artifacts root:",
        f"#   {paths.artifacts}",
        "",
    ]
    for key, value in values.items():
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif value is None:
            text = "null"
        else:
            text = str(value)
        lines.append(f"{key}: {text}")
    return "\n".join(lines) + "\n"


def write_config(paths, profile: Profile, run: str, **overrides) -> "Path":
    from pathlib import Path

    paths.config_dir.mkdir(parents=True, exist_ok=True)
    path = Path(paths.config_path(run))
    path.write_text(render_config(profile, run, paths, **overrides), encoding="utf-8", newline="\n")
    return path
