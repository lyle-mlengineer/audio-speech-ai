"""Where everything lives.

The rule: **anything expensive must be reachable after the VM dies.** On Colab
that means the artifacts root is a Google Drive folder; locally it defaults to
``<repo>/artifacts``.

Resolution order (first match wins):

1. ``--artifacts PATH`` (explicit flag)
2. ``$SHENG_CHATTERBOX_ARTIFACTS``
3. ``/content/drive/MyDrive/chatterbox-sheng`` when Drive is mounted
4. ``<repo>/artifacts``

Pure stdlib, importable on a bare Colab VM before any ML dependency exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------- constants
DATASET_ID = "lyle-mlengineer/kenyan-celebs"
SPEAKER = "abel"

MODEL_REPO_ID = "ResembleAI/chatterbox"
# Multilingual T3 (Llama-520M). "sw" (Swahili) is in SUPPORTED_LANGUAGES.
MTL_T3_MODEL = "t3_mtl23ls_v2.safetensors"
# Files the multilingual loader needs; everything else in the repo is optional.
PRETRAINED_FILES = [
    "ve.pt",
    MTL_T3_MODEL,
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
]

# S3 tokenizer / voice encoder input rate. Chatterbox also synthesises at
# S3GEN_SR=24000, but we never touch s3gen so 16 kHz is all we need on disk.
SAMPLE_RATE = 16000
LANGUAGE_ID = "sw"

DEFAULT_RUN = "sheng_t3"
DEFAULT_STEPS = 600

ENV_VAR = "SHENG_CHATTERBOX_ARTIFACTS"
DRIVE_DEFAULT = Path("/content/drive/MyDrive/chatterbox-sheng")

REPO_ROOT = Path(__file__).resolve().parents[2]


def colab_drive_mounted() -> bool:
    """True when Google Drive is already mounted (cheap, no import of drive)."""
    return Path("/content/drive/MyDrive").is_dir()


def on_drive(path: Path) -> bool:
    p = str(path).replace("\\", "/")
    return p.startswith("/content/drive/")


def resolve_artifacts(explicit: str | os.PathLike | None = None) -> Path:
    """Pick the artifacts root, see module docstring for the order."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    if colab_drive_mounted():
        return DRIVE_DEFAULT.resolve()
    return (REPO_ROOT / "artifacts").resolve()


# ------------------------------------------------------------------------ layout
@dataclass(frozen=True)
class Paths:
    artifacts: Path

    # ---- inputs ---------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.artifacts / "data"

    @property
    def wav_dir(self) -> Path:
        return self.data_dir / "wav16k"

    @property
    def dataset_info(self) -> Path:
        return self.data_dir / "dataset_info.json"

    @property
    def data_report(self) -> Path:
        return self.data_dir / "data_report.md"

    @property
    def manifest_dir(self) -> Path:
        return self.artifacts / "manifests"

    @property
    def train_manifest(self) -> Path:
        return self.manifest_dir / "train.jsonl"

    @property
    def val_manifest(self) -> Path:
        return self.manifest_dir / "val.jsonl"

    # ---- model ----------------------------------------------------------
    @property
    def pretrained_dir(self) -> Path:
        """Local copy of the HF snapshot (`ve.pt`, `t3_*.safetensors`, ...)."""
        return self.artifacts / "pretrained" / "chatterbox"

    @property
    def base_t3_path(self) -> Path:
        return self.pretrained_dir / MTL_T3_MODEL

    # ---- runs -----------------------------------------------------------
    @property
    def config_dir(self) -> Path:
        return self.artifacts / "configs"

    def config_path(self, run: str) -> Path:
        return self.config_dir / f"{run}.yaml"

    @property
    def checkpoint_root(self) -> Path:
        return self.artifacts / "checkpoints"

    def checkpoint_dir(self, run: str) -> Path:
        return self.checkpoint_root / run

    def step_dir(self, run: str, step: int) -> Path:
        return self.checkpoint_dir(run) / f"step_{step:07d}"

    def latest_dir(self, run: str) -> Path:
        return self.checkpoint_dir(run) / "latest"

    @property
    def log_root(self) -> Path:
        return self.artifacts / "logs"

    def log_dir(self, run: str) -> Path:
        return self.log_root / run

    def console_log(self, run: str) -> Path:
        return self.log_dir(run) / "console.log"

    def train_log(self, run: str) -> Path:
        """The trainer's own log (stdout+stderr), tailed for heartbeat/step."""
        return self.log_dir(run) / "train.log"

    def eval_dir(self, run: str) -> Path:
        return self.artifacts / "evals" / run

    def export_dir(self, run: str) -> Path:
        return self.artifacts / "exports" / run

    # ---- feature cache --------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.artifacts / "cache"

    def feature_dir(self, tag: str) -> Path:
        return self.cache_dir / f"features_{tag}"

    # ---- progress -------------------------------------------------------
    @property
    def progress_dir(self) -> Path:
        return self.artifacts / "progress"

    @property
    def state_json(self) -> Path:
        return self.progress_dir / "state.json"

    @property
    def runs_jsonl(self) -> Path:
        return self.progress_dir / "runs.jsonl"

    @property
    def repo_progress_md(self) -> Path:
        return REPO_ROOT / "progress.md"

    @property
    def artifacts_progress_md(self) -> Path:
        return self.progress_dir / "progress.md"

    # ---- helpers --------------------------------------------------------
    def ensure(self) -> "Paths":
        for p in (
            self.data_dir,
            self.wav_dir,
            self.manifest_dir,
            self.pretrained_dir,
            self.config_dir,
            self.checkpoint_root,
            self.log_root,
            self.cache_dir,
            self.progress_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def rel_to_manifest(self, path: Path) -> str:
        """Path as stored inside a manifest: relative to ``manifests/``."""
        return os.path.relpath(Path(path).resolve(), self.manifest_dir.resolve()).replace("\\", "/")

    def rel(self, path: Path) -> str:
        """Short label for logs: path relative to the artifacts root when possible."""
        try:
            return str(Path(path).resolve().relative_to(self.artifacts.resolve()))
        except ValueError:
            return str(path)


def latest_checkpoint(paths: Paths, run: str) -> Path | None:
    """``latest`` if present, else the highest ``step_*`` dir, else None."""
    latest = paths.latest_dir(run)
    if (latest / "t3_cfg.safetensors").exists():
        return latest
    ckpt_dir = paths.checkpoint_dir(run)
    if not ckpt_dir.is_dir():
        return None
    steps = sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_"))
    for step in reversed(steps):
        if (step / "t3_cfg.safetensors").exists():
            return step
    return None


def checkpoint_step(path: Path) -> int | None:
    """Read the step number out of ``step_0000420``; None for ``latest``."""
    name = Path(path).name
    if name.startswith("step_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None
