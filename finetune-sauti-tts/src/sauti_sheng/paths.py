"""Where everything lives.

The rule: **anything expensive must be reachable after the VM dies.** On Colab
that means the artifacts root is a Google Drive folder; locally it defaults to
``<repo>/artifacts``.

Resolution order (first match wins):

1. ``--artifacts PATH`` (explicit flag)
2. ``$SAUTI_SHENG_ARTIFACTS``
3. ``/content/drive/MyDrive/sauti-sheng`` when Drive is mounted
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

#: MsingiAI's Swahili TTS release. It is an F5-TTS v1 Base fine-tune, shipped as
#: a *full training checkpoint* (model + EMA + optimizer + scheduler + update).
MODEL_REPO_ID = "msingiai/sauti-tts"
MODEL_FILE = "model_last.pt"
VOCAB_FILE = "vocab.txt"

#: F5-TTS is 24 kHz-only: ``MelSpec``, the Vocos vocoder and the training data
#: loader all assume it.
SAMPLE_RATE = 24000
#: The parquet actually holds mono 22.05 kHz audio. The dataset card says
#: 16 kHz; it is wrong (checked with soundfile on all 42 rows).
SOURCE_SAMPLE_RATE = 22050

DEFAULT_RUN = "sheng_abel"

ENV_VAR = "SAUTI_SHENG_ARTIFACTS"
SCRATCH_ENV_VAR = "SAUTI_SHENG_SCRATCH"
DRIVE_DEFAULT = Path("/content/drive/MyDrive/sauti-sheng")

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


def resolve_scratch(explicit: str | os.PathLike | None = None) -> Path:
    """Fast local disk for the 5.4 GB raw checkpoint and the parquet.

    Never put the raw checkpoint on Drive if you can avoid it: it is deleted
    after the EMA weights are extracted.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(SCRATCH_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    if Path("/content").is_dir() and os.access("/content", os.W_OK):
        return Path("/content/sauti-sheng-scratch")
    return resolve_artifacts() / ".scratch"


# ------------------------------------------------------------------------ layout
@dataclass(frozen=True)
class Paths:
    artifacts: Path
    scratch: Path | None = None
    #: where progress.md lives; tests point this at a tmp file so they never
    #: rewrite the real one.
    repo_progress: Path | None = None

    # ---- inputs ---------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.artifacts / "data"

    @property
    def wav_dir(self) -> Path:
        """24 kHz mono WAVs, the only form F5-TTS training accepts."""
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
    def train_csv(self) -> Path:
        return self.data_dir / "metadata_train.csv"

    @property
    def val_csv(self) -> Path:
        return self.data_dir / "metadata_val.csv"

    @property
    def ref_audio(self) -> Path:
        return self.data_dir / "ref_audio.wav"

    @property
    def ref_text_file(self) -> Path:
        return self.data_dir / "ref_text.txt"

    @property
    def ref_meta(self) -> Path:
        return self.data_dir / "ref_meta.json"

    # ---- F5-TTS dataset -------------------------------------------------
    @property
    def f5_dataset_dir(self) -> Path:
        return self.data_dir / "f5"

    @property
    def f5_raw_arrow(self) -> Path:
        return self.f5_dataset_dir / "raw.arrow"

    @property
    def f5_duration_json(self) -> Path:
        return self.f5_dataset_dir / "duration.json"

    @property
    def f5_vocab(self) -> Path:
        return self.f5_dataset_dir / "vocab.txt"

    # ---- model ----------------------------------------------------------
    @property
    def pretrained_dir(self) -> Path:
        return self.artifacts / "pretrained" / "sauti-tts"

    @property
    def sauti_raw_ckpt(self) -> Path:
        """The 5.39 GB release file; lives on scratch, deleted after stripping."""
        return (self.scratch or self.pretrained_dir) / MODEL_FILE

    @property
    def sauti_ema_ckpt(self) -> Path:
        """EMA-only checkpoint (``ema_model_state_dict``), ~1.35 GB, Drive-safe."""
        return self.pretrained_dir / "sauti_ema.pt"

    @property
    def sauti_vocab(self) -> Path:
        return self.pretrained_dir / VOCAB_FILE

    @property
    def model_report(self) -> Path:
        return self.pretrained_dir / "model_report.md"

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

    def pretrained_pointer(self, run: str) -> Path:
        """``pretrained_sauti.pt`` inside the run dir.

        F5's ``Trainer`` treats a directory containing only ``pretrained_*`` as
        "initialise from this" and a directory containing ``model_last.pt`` as
        "resume". The name is what separates the two, so it matters.
        """
        return self.checkpoint_dir(run) / "pretrained_sauti.pt"

    def latest_checkpoint(self, run: str) -> Path:
        return self.checkpoint_dir(run) / "model_last.pt"

    @property
    def log_root(self) -> Path:
        return self.artifacts / "logs"

    def log_dir(self, run: str) -> Path:
        return self.log_root / run

    def console_log(self, run: str) -> Path:
        return self.log_dir(run) / "console.log"

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
            self.f5_dataset_dir,
            self.pretrained_dir,
            self.config_dir,
            self.checkpoint_root,
            self.progress_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
        if self.scratch is not None:
            self.scratch.mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, path: Path) -> str:
        """Short label for logs: path relative to the artifacts root when possible."""
        try:
            return str(Path(path).resolve().relative_to(self.artifacts.resolve()))
        except ValueError:
            return str(path)


# ------------------------------------------------------------------- checkpoints
def is_f5_checkpoint(path: Path) -> bool:
    """F5 ``Trainer`` checkpoints are a single ``.pt`` file, not a directory."""
    return Path(path).is_file() and Path(path).suffix in {".pt", ".safetensors"}


def checkpoint_update(path: Path) -> int | None:
    """Read the update counter out of ``model_1234.pt``; None for ``model_last``."""
    name = Path(path).name
    if not name.startswith("model_") or name == "model_last.pt":
        return None
    try:
        return int(name.split("_", 1)[1].split(".")[0])
    except ValueError:
        return None


def all_checkpoints(paths: Paths, run: str) -> list[Path]:
    ckpt_dir = paths.checkpoint_dir(run)
    if not ckpt_dir.is_dir():
        return []
    return sorted(
        (
            p
            for p in ckpt_dir.iterdir()
            if p.is_file()
            and p.name.startswith("model_")
            and p.name != "model_last.pt"
            and p.suffix == ".pt"
        ),
        key=lambda p: checkpoint_update(p) or 0,
    )
