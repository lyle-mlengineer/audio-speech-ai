"""Where everything lives.

The rule: **anything expensive must be reachable after the VM dies.** On Colab
that means the artifacts root is a Google Drive folder; locally it defaults to
``<repo>/artifacts``.

Resolution order (first match wins):

1. ``--artifacts PATH`` (explicit flag)
2. ``$SHENG_QWENTTS_ARTIFACTS``
3. ``/content/drive/MyDrive/qwentts-sheng`` when Drive is mounted
4. ``<repo>/artifacts``

Pure stdlib, importable on a bare Colab VM before any ML dependency exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------- constants
DATASET_ID = "lyle-mlengineer/kenyan-celebs"
DATASET_PARQUET = "data/train-00000-of-00001.parquet"
SPEAKER = "abel"  # every row of the dataset is this one speaker

MODEL_REPO_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TOKENIZER_REPO_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

# The tokenizer (and the speaker encoder + the training mel) are 24 kHz-only:
# dataset.py in the upstream repo asserts sr == 24000.
SAMPLE_RATE = 24000
# The parquet holds mono 16-bit WAV at 22.05 kHz (the card says 16 kHz - it lies).
SOURCE_SAMPLE_RATE = 22050

DEFAULT_RUN = "sheng_06b"
DEFAULT_STEPS = 240

ENV_VAR = "SHENG_QWENTTS_ARTIFACTS"
DRIVE_DEFAULT = Path("/content/drive/MyDrive/qwentts-sheng")

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
    #: where progress.md lives; tests point this at a tmp file so they never
    #: rewrite the real one.
    repo_progress: Path | None = None

    # ---- inputs ---------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.artifacts / "data"

    @property
    def wav_dir(self) -> Path:
        """24 kHz mono WAVs, the only form the Qwen3-TTS tooling accepts."""
        return self.data_dir / "wav24k"

    @property
    def raw_parquet(self) -> Path:
        return self.data_dir / "kenyan-celebs.parquet"

    @property
    def dataset_info(self) -> Path:
        return self.data_dir / "dataset_info.json"

    @property
    def data_report(self) -> Path:
        return self.data_dir / "data_report.md"

    @property
    def ref_audio(self) -> Path:
        return self.data_dir / "ref_audio.wav"

    @property
    def manifest_dir(self) -> Path:
        return self.artifacts / "manifests"

    @property
    def train_raw_jsonl(self) -> Path:
        return self.manifest_dir / "train_raw.jsonl"

    @property
    def val_raw_jsonl(self) -> Path:
        return self.manifest_dir / "val_raw.jsonl"

    @property
    def train_codes_jsonl(self) -> Path:
        return self.manifest_dir / "train_with_codes.jsonl"

    @property
    def val_codes_jsonl(self) -> Path:
        return self.manifest_dir / "val_with_codes.jsonl"

    # ---- model ----------------------------------------------------------
    @property
    def pretrained_root(self) -> Path:
        return self.artifacts / "pretrained"

    @property
    def model_dir(self) -> Path:
        """Local snapshot of Qwen3-TTS-12Hz-0.6B-Base (needs to be a real dir:
        the trainer copies it for every checkpoint)."""
        return self.pretrained_root / "qwen3-tts-0.6b-base"

    @property
    def tokenizer_dir(self) -> Path:
        return self.pretrained_root / "qwen3-tts-tokenizer-12hz"

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

    def latest_pointer(self, run: str) -> Path:
        """Tiny file holding the name of the newest usable ``step_*`` directory.

        A real ``latest/`` directory would mean copying ~1.4 GB on every save;
        with Google Drive that is minutes per checkpoint. One line of text is
        not.
        """
        return self.checkpoint_dir(run) / "latest.txt"

    @property
    def log_root(self) -> Path:
        return self.artifacts / "logs"

    def log_dir(self, run: str) -> Path:
        return self.log_root / run

    def console_log(self, run: str) -> Path:
        return self.log_dir(run) / "console.log"

    def train_log(self, run: str) -> Path:
        return self.log_dir(run) / "train.log"

    def eval_dir(self, run: str) -> Path:
        return self.artifacts / "evals" / run

    def export_dir(self, run: str) -> Path:
        return self.artifacts / "exports" / run

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
        return self.repo_progress or (REPO_ROOT / "progress.md")

    @property
    def artifacts_progress_md(self) -> Path:
        return self.progress_dir / "progress.md"

    # ---- helpers --------------------------------------------------------
    def ensure(self) -> "Paths":
        for p in (
            self.data_dir,
            self.wav_dir,
            self.manifest_dir,
            self.pretrained_root,
            self.config_dir,
            self.checkpoint_root,
            self.log_root,
            self.progress_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, path: Path) -> str:
        """Short label for logs: path relative to the artifacts root when possible."""
        try:
            return str(Path(path).resolve().relative_to(self.artifacts.resolve()))
        except ValueError:
            return str(path)


# ------------------------------------------------------------------- checkpoints
STATE_FILE = "train_state.json"
SPEAKER_FILE = "speaker_embedding.pt"
WEIGHTS_FILE = "model.safetensors"


def is_checkpoint(path: Path) -> bool:
    return (Path(path) / STATE_FILE).is_file() and (Path(path) / WEIGHTS_FILE).is_file()


def checkpoint_step(path: Path) -> int | None:
    """Read the step number out of ``step_0000040``; None for ``latest``."""
    name = Path(path).name
    if name.startswith("step_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def latest_checkpoint(paths: Paths, run: str) -> Path | None:
    """Follow ``latest.txt`` when it points at a real checkpoint, else the highest ``step_*``."""
    ckpt_dir = paths.checkpoint_dir(run)
    pointer = paths.latest_pointer(run)
    if pointer.is_file():
        name = pointer.read_text(encoding="utf-8").strip()
        if name:
            target = ckpt_dir / name
            if is_checkpoint(target):
                return target
    if not ckpt_dir.is_dir():
        return None
    steps = sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_"))
    for step in reversed(steps):
        if is_checkpoint(step):
            return step
    return None


def all_checkpoints(paths: Paths, run: str) -> list[Path]:
    ckpt_dir = paths.checkpoint_dir(run)
    if not ckpt_dir.is_dir():
        return []
    return sorted(
        (p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_") and is_checkpoint(p)),
        key=lambda p: checkpoint_step(p) or 0,
    )
