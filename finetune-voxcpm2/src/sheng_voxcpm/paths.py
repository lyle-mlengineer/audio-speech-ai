"""Path layout for the Sheng x VoxCPM2 pipeline.

Single source of truth: every module derives its paths from :class:`Paths`.

Resolution order for the artifacts root (where all heavy/expensive state lives):

1. ``--artifacts`` CLI flag
2. ``SHENG_ARTIFACTS`` environment variable
3. ``/content/drive/MyDrive/voxcpm2-sheng`` when a Colab Drive is mounted
4. ``<repo>/artifacts`` otherwise

Rule of thumb: anything that costs GPU time or a download lives under
``artifacts``; only code lives in the git repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- fixed identifiers -------------------------------------------------
DATASET_ID = "lyle-mlengineer/kenyan-celebs"
BASE_MODEL_ID = "openbmb/VoxCPM2"
VOXCPM_REPO_URL = "https://github.com/OpenBMB/VoxCPM.git"
VOXCPM_REPO_NAME = "VoxCPM"

DRIVE_DEFAULT = Path("/content/drive/MyDrive/voxcpm2-sheng")
TARGET_SAMPLE_RATE = 16_000  # AudioVAE encoder input; MUST match the training config
OUT_SAMPLE_RATE = 48_000  # AudioVAE decoder output


def repo_root() -> Path:
    """Root of this project (the directory holding README.md / src / configs)."""
    return Path(__file__).resolve().parents[2]


def _drive_available() -> bool:
    return Path("/content/drive/MyDrive").is_dir()


@dataclass(frozen=True)
class Paths:
    artifacts: Path
    code_root: Path

    # ---------------------------------------------------------------- setup
    @classmethod
    def resolve(cls, artifacts: str | os.PathLike | None = None) -> "Paths":
        if artifacts:
            root = Path(artifacts).expanduser()
        elif os.environ.get("SHENG_ARTIFACTS"):
            root = Path(os.environ["SHENG_ARTIFACTS"]).expanduser()
        elif _drive_available():
            root = DRIVE_DEFAULT
        else:
            root = repo_root() / "artifacts"
        return cls(artifacts=root.resolve(), code_root=repo_root())

    def ensure(self) -> "Paths":
        for d in (
            self.artifacts,
            self.data_dir,
            self.wav_dir,
            self.manifest_dir,
            self.config_dir,
            self.checkpoint_root,
            self.log_root,
            self.eval_root,
            self.progress_dir,
            self.pretrained_parent,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # ------------------------------------------------------------- sources
    @property
    def voxcpm_repo(self) -> Path:
        """Cloned OpenBMB/VoxCPM repo (contains scripts/train_voxcpm_finetune.py)."""
        override = os.environ.get("VOXCPM_REPO")
        return Path(override).expanduser() if override else self.artifacts / "repo" / VOXCPM_REPO_NAME

    @property
    def pretrained_parent(self) -> Path:
        return self.artifacts / "pretrained_models"

    @property
    def model_dir(self) -> Path:
        """Local snapshot of openbmb/VoxCPM2 (a *directory*, not a Hub id)."""
        return self.pretrained_parent / "VoxCPM2"

    @property
    def base_config_json(self) -> Path:
        return self.model_dir / "config.json"

    # ---------------------------------------------------------------- data
    @property
    def data_dir(self) -> Path:
        return self.artifacts / "data"

    @property
    def wav_dir(self) -> Path:
        """16 kHz mono WAVs, one file per clip, named ``<speaker>_<idx:04d>.wav``."""
        return self.data_dir / "wav16k"

    @property
    def manifest_dir(self) -> Path:
        return self.artifacts / "manifests"

    @property
    def train_manifest(self) -> Path:
        return self.manifest_dir / "train.jsonl"

    @property
    def val_manifest(self) -> Path:
        return self.manifest_dir / "val.jsonl"

    @property
    def data_stats(self) -> Path:
        return self.manifest_dir / "dataset_stats.json"

    @property
    def data_report(self) -> Path:
        return self.manifest_dir / "data_report.md"

    # ------------------------------------------------------------- configs
    @property
    def config_dir(self) -> Path:
        return self.artifacts / "configs"

    def template_dir(self) -> Path:
        return self.code_root / "configs"

    # --------------------------------------------------------- checkpoints
    @property
    def checkpoint_root(self) -> Path:
        return self.artifacts / "checkpoints"

    def checkpoint_dir(self, run: str) -> Path:
        return self.checkpoint_root / run

    @property
    def log_root(self) -> Path:
        return self.artifacts / "logs"

    def log_dir(self, run: str) -> Path:
        return self.log_root / run

    def train_log(self, run: str) -> Path:
        return self.log_dir(run) / "train.log"

    def tensorboard_dir(self, run: str) -> Path:
        return self.log_dir(run) / "tensorboard"

    # ---------------------------------------------------------------- eval
    @property
    def eval_root(self) -> Path:
        return self.artifacts / "evals"

    def eval_dir(self, run: str) -> Path:
        return self.eval_root / run

    # ------------------------------------------------------------ progress
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
    def artifacts_progress_md(self) -> Path:
        return self.progress_dir / "progress.md"

    @property
    def repo_progress_md(self) -> Path:
        return self.code_root / "progress.md"

    # -------------------------------------------------------------- helpers
    def rel_to_manifest(self, wav: Path) -> str:
        """Manifest audio paths are resolved relative to the manifest's own dir."""
        return Path(os.path.relpath(wav, self.manifest_dir)).as_posix()

    def describe(self) -> str:
        return f"artifacts={self.artifacts}"
