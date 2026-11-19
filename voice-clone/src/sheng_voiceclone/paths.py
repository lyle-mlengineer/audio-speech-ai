"""Where everything lives.

Rule of this project: **measurement lives next to the artifact it measures.**
Every stage writes into the artifacts root, so an eval, a score and the audio it
scored always sit in one directory and can be diffed later.

Resolution order for the artifacts root (first match wins):

1. ``--artifacts PATH`` (explicit flag)
2. ``$SHENG_VOICECLONE_ARTIFACTS``
3. ``/content/drive/MyDrive/voice-clone-sheng`` when Drive is mounted
4. ``<repo>/voice-clone/artifacts``

Pure stdlib, so ``sheng-voiceclone status`` works on a bare Colab VM right after a
disconnect, before any ML dependency exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------- dataset
DATASET_ID = "lyle-mlengineer/kenyan-celebs"
DATASET_PARQUET = "data/train-00000-of-00001.parquet"
SPEAKER = "abel"

# ----------------------------------------------------------------------- audio
# Chatterbox's voice encoder, S3 tokenizer and training featuriser all consume
# 16 kHz mono; the vocoder synthesises at 24 kHz. We only ever store 16 kHz.
SAMPLE_RATE = 16000
SOURCE_SAMPLE_RATE = 22050

# ----------------------------------------------------------------------- model
# Chatterbox Multilingual: "sw" (Swahili) is in SUPPORTED_LANGUAGES, so the base
# model already clones any voice from a ~6 s reference clip, in Swahili.
MODEL_REPO_ID = "ResembleAI/chatterbox"
LANGUAGE_ID = "sw"

# ------------------------------------------------------------------ run naming
DEFAULT_RUN = "sheng_clone"
ENV_VAR = "SHENG_VOICECLONE_ARTIFACTS"
DRIVE_DEFAULT = Path("/content/drive/MyDrive/voice-clone-sheng")

REPO_ROOT = Path(__file__).resolve().parents[2]


def on_drive(path: Path) -> bool:
    """True when the path lives under a mounted Google Drive."""
    return str(path).replace("\\", "/").startswith("/content/drive/")


def is_inside(child: Path, parent: Path) -> bool:
    """True when ``child`` is ``parent`` or below it (no filesystem access)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def colab_drive_mounted() -> bool:
    """True when Google Drive is already mounted (no drive import needed)."""
    return Path("/content/drive/MyDrive").is_dir()


@dataclass
class Paths:
    """Artifacts root plus every derived location, as plain attributes."""

    root: Path

    # ---------------------------------------------------------------- derived
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def wav_dir(self) -> Path:
        return self.data_dir / "wav16k"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def train_manifest(self) -> Path:
        return self.manifests / "train.jsonl"

    @property
    def val_manifest(self) -> Path:
        return self.manifests / "val.jsonl"

    @property
    def test_manifest(self) -> Path:
        return self.manifests / "test.jsonl"

    @property
    def ref_audio(self) -> Path:
        """The clip used to clone the voice (held out of training)."""
        return self.data_dir / "ref_audio.wav"

    @property
    def ref_text(self) -> Path:
        return self.data_dir / "ref_text.txt"

    @property
    def pretrained_dir(self) -> Path:
        return self.root / "pretrained" / "chatterbox"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def evals(self) -> Path:
        return self.root / "evals"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def progress_dir(self) -> Path:
        return self.root / "progress"

    @property
    def state_file(self) -> Path:
        return self.progress_dir / "state.json"

    @property
    def events_file(self) -> Path:
        return self.progress_dir / "runs.jsonl"

    # ------------------------------------------------------------------ setup
    def ensure_dirs(self) -> None:
        for p in (
            self.root,
            self.data_dir,
            self.wav_dir,
            self.raw_dir,
            self.manifests,
            self.checkpoints,
            self.evals,
            self.scores,
            self.configs,
            self.logs,
            self.progress_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path) -> str:
        """Path relative to the artifacts root, POSIX-style, for reports."""
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except (ValueError, OSError):
            return str(path)


def default_root() -> Path:
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    if colab_drive_mounted():
        return DRIVE_DEFAULT
    return REPO_ROOT / "artifacts"


def resolve(artifacts: str | None = None) -> Paths:
    """Build ``Paths`` from an explicit flag, the env var, Drive, or the repo."""
    root = Path(artifacts).expanduser() if artifacts else default_root()
    return Paths(root=root)
