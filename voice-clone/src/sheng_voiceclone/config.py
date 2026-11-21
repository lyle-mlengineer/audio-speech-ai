"""Training profiles, rendered to YAML without a yaml dependency.

Deep-merges onto the same numbers the sibling ``finetune-chatterbox`` project
uses, so a run started here and a run started there mean the same thing:

| profile | GPU | batch x accum | dtype |
|---|---|---|---|
| l4   | Colab L4 24 GB | 4 x 2 | bfloat16 |
| a100 | Colab A100 40/80 GB | 8 x 1 | bfloat16 |
| t4   | Colab T4 16 GB | 2 x 4 | float16 |

Effective batch is 8 everywhere, so ``steps`` and ``lr`` mean the same thing on
every GPU and a run can be resumed on a different one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import Paths

EFFECTIVE_BATCH = 8


@dataclass(frozen=True)
class Profile:
    name: str
    batch_size: int
    grad_accum: int
    dtype: str
    # 200 steps x batch 8 = 1600 clips ~= 47 passes over the ~17 min train split.
    # 600 (the old default) was ~140 epochs: well into the overfit zone for a
    # voice fine-tune; the scoring harness penalises the word loss it causes.
    steps: int = 200
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    save_interval: int = 100
    keep_last: int = 3
    log_interval: int = 10
    min_vram_gb: float = 15.0
    allow_small_gpu: bool = False
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


PROFILES: dict[str, Profile] = {
    "l4": Profile("l4", batch_size=4, grad_accum=2, dtype="bfloat16", min_vram_gb=20.0),
    "a100": Profile("a100", batch_size=8, grad_accum=1, dtype="bfloat16", min_vram_gb=38.0),
    "t4": Profile(
        "t4", batch_size=2, grad_accum=4, dtype="float16", min_vram_gb=14.0, allow_small_gpu=True
    ),
}


def profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile '{name}' (have: {', '.join(PROFILES)})") from None


def pick_profile(vram_gb: float) -> str:
    """A GPU's VRAM -> the profile that will not OOM (conservative, largest fits)."""
    if vram_gb >= 38.0:
        return "a100"
    if vram_gb >= 20.0:
        return "l4"
    return "t4"


def override(prof: Profile, **changes: Any) -> Profile:
    """``--steps``/``--lr``/``--lora`` flags on top of a profile."""
    fields = {k: v for k, v in changes.items() if v is not None}
    if not fields:
        return prof
    unknown = set(fields) - set(asdict(prof))
    if unknown:
        raise KeyError(f"unknown profile field(s): {', '.join(sorted(unknown))}")
    data = asdict(prof) | fields
    return Profile(**data)


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def render(prof: Profile, paths: Paths, *, run: str) -> str:
    """Resolve a profile against an artifacts root into the run's YAML."""
    lines = [
        f"# rendered by sheng-voiceclone (profile: {prof.name}) - do not edit by hand",
        f"# artifacts root: {paths.root}",
        f"run: {run}",
        f"profile: {prof.name}",
        f"effective_batch: {prof.effective_batch}",
    ]
    for k, v in asdict(prof).items():
        if k == "name":
            continue
        lines.append(f"{k}: {_yaml_scalar(v)}")
    lines += [
        f"train_manifest: {paths.train_manifest}",
        f"val_manifest: {paths.val_manifest}",
        f"ref_audio: {paths.ref_audio}",
        f"pretrained_dir: {paths.pretrained_dir}",
        f"save_dir: {paths.checkpoints / run}",
        f"eval_dir: {paths.evals / run}",
    ]
    return "\n".join(lines) + "\n"


def write_yaml(prof: Profile, paths: Paths, *, run: str) -> Path:
    paths.ensure_dirs()
    out = paths.configs / f"{run}.yaml"
    out.write_text(render(prof, paths, run=run), encoding="utf-8")
    return out


def parse_yaml(text: str) -> dict[str, Any]:
    """Parse the flat ``key: value`` YAML this module writes (nothing else)."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        raw = raw.strip()
        if raw.lower() in {"true", "false"}:
            out[key.strip()] = raw.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    out[key.strip()] = cast(raw)
                    break
                except ValueError:
                    continue
            else:
                out[key.strip()] = raw.strip('"')
    return out
