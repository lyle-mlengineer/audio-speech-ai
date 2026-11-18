"""Path layout and fixed identifiers.

Everything that costs GPU time or a download lives under the **artifacts root**
so a Colab disconnect costs at most one ``save_interval`` of training.

Resolution order for the artifacts root:

1. ``--artifacts PATH``
2. ``$KENYA_ARTIFACTS``
3. ``/content/drive/MyDrive/kenya-voices`` when Drive is mounted
4. ``<repo>/artifacts``
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_MODEL_ID = "openbmb/VoxCPM2"  # Apache-2.0, 30 languages incl. Swahili, voice design + LoRA
VOXCPM_REPO_URL = "https://github.com/OpenBMB/VoxCPM.git"
VOXCPM_REPO_NAME = "VoxCPM"

DRIVE_DEFAULT = Path("/content/drive/MyDrive/kenya-voices")
TARGET_SAMPLE_RATE = 16_000  # AudioVAE encoder input (asserted by the trainer)
OUT_SAMPLE_RATE = 48_000  # AudioVAE decoder output
DEFAULT_RUN = "design_r32"

# Uniquely Kenyan / African sources. The first is the workhorse (68k clips,
# 385 speakers, 16 kHz, transcripts + gender/age metadata).
DATASET_PRESETS: dict[str, dict[str, str]] = {
    # 64.5k clips, ~380 Kenyan speakers, 16 kHz, gender/age/location metadata.
    # The workhorse: many speakers is what a voice-design adapter needs.
    "badrex": {
        "repo": "badrex/swahili-speech-400hr",
        "split": "train",
        "audio": "audio",
        "text": "transcription",
        "speaker": "voice_creator_id",
        "id": "audio_id",
        "duration": "audio_duration",
        "gender": "gender",
        "age": "age_group",
        "location": "location",
        "language": "Swahili",
        "license": "CC-BY-4.0",
    },
    # A small custom set (audio + text only). No speaker/gender/age columns: every
    # clip is attributed to one speaker, so this can shift the accent but cannot
    # teach "design a voice". Use it for a quick sanity run, not for the real adapter.
    "kenyan-voices": {
        "repo": "lyle-mlengineer/kenyan-voices",
        "split": "train",
        "audio": "audio",
        "text": "text",
        "speaker": "",
        "id": "",
        "duration": "",
        "gender": "",
        "age": "",
        "location": "",
        "language": "Swahili",
        "license": "see dataset card",
    },
}
DEFAULT_DATASETS = ("badrex",)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _drive_available() -> bool:
    return Path("/content/drive/MyDrive").is_dir()


@dataclass(frozen=True)
class Paths:
    artifacts: Path
    code_root: Path

    @classmethod
    def resolve(cls, artifacts: str | os.PathLike | None = None) -> "Paths":
        if artifacts:
            root = Path(artifacts).expanduser()
        elif os.environ.get("KENYA_ARTIFACTS"):
            root = Path(os.environ["KENYA_ARTIFACTS"]).expanduser()
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
            self.meta_dir,
            self.manifest_dir,
            self.config_dir,
            self.progress_dir,
            self.checkpoint_root,
            self.log_root,
            self.eval_root,
            self.export_root,
            self.pretrained_parent,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # ------------------------------------------------------------- sources
    @property
    def voxcpm_repo(self) -> Path:
        override = os.environ.get("VOXCPM_REPO")
        return Path(override).expanduser() if override else self.artifacts / "repo" / VOXCPM_REPO_NAME

    @property
    def train_script(self) -> Path:
        return self.voxcpm_repo / "scripts" / "train_voxcpm_finetune.py"

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
    def meta_dir(self) -> Path:
        """Per-speaker metadata cache (selected rows, measured features)."""
        return self.data_dir / "meta"

    @property
    def speaker_profiles(self) -> Path:
        """Speaker-level caption attributes: gender, age band, pitch, pace, location."""
        return self.meta_dir / "speakers.json"

    @property
    def excluded_speakers(self) -> Path:
        """Speaker ids held out entirely (val) - never trained on."""
        return self.meta_dir / "val_speakers.json"

    @property
    def data_report(self) -> Path:
        return self.manifest_dir / "data_report.md"

    # ------------------------------------------------------------- configs
    @property
    def config_dir(self) -> Path:
        return self.artifacts / "configs"

    def run_config(self, run: str) -> Path:
        return self.config_dir / f"{run}.yaml"

    # --------------------------------------------------------- checkpoints
    @property
    def checkpoint_root(self) -> Path:
        return self.artifacts / "checkpoints"

    def checkpoint_dir(self, run: str) -> Path:
        return self.checkpoint_root / run

    @property
    def log_root(self) -> Path:
        return self.artifacts / "logs"

    def console_log(self, run: str) -> Path:
        return self.log_root / run / "console.log"

    def train_log(self, run: str) -> Path:
        return self.log_root / run / "train.log"

    def tensorboard_dir(self, run: str) -> Path:
        return self.log_root / run / "tensorboard"

    def trainer_log(self, run: str) -> Path:
        """``save_path/train.log`` - written by the upstream trainer itself."""
        return self.checkpoint_dir(run) / "train.log"

    # ---------------------------------------------------------------- eval
    @property
    def eval_root(self) -> Path:
        return self.artifacts / "evals"

    def eval_dir(self, run: str) -> Path:
        return self.eval_root / run

    # -------------------------------------------------------------- export
    @property
    def export_root(self) -> Path:
        return self.artifacts / "exports"

    def export_dir(self, run: str) -> Path:
        return self.export_root / run

    # ------------------------------------------------------------- progress
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
        return self.code_root / "progress.md"

    @property
    def artifacts_progress_md(self) -> Path:
        return self.artifacts / "progress.md"

    # -------------------------------------------------------------- helpers
    def rel_to_manifest(self, wav: Path) -> str:
        return Path(os.path.relpath(wav, self.manifest_dir)).as_posix()

    def describe(self) -> str:
        return f"artifacts={self.artifacts} code={self.code_root}"
