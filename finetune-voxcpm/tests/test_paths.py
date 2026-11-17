"""Artifacts-root resolution and the derived path layout."""

from __future__ import annotations

from pathlib import Path

from sheng_voxcpm import paths as P


def test_explicit_artifacts_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("SHENG_ARTIFACTS", str(tmp_path / "from-env"))
    resolved = P.Paths.resolve(tmp_path / "explicit")
    assert resolved.artifacts == (tmp_path / "explicit").resolve()
    assert resolved.code_root == P.repo_root()


def test_env_var_when_no_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("SHENG_ARTIFACTS", str(tmp_path / "from-env"))
    assert P.Paths.resolve().artifacts == (tmp_path / "from-env").resolve()


def test_repo_artifacts_without_env_or_drive(monkeypatch):
    monkeypatch.delenv("SHENG_ARTIFACTS", raising=False)
    monkeypatch.setattr(P, "_drive_available", lambda: False)
    assert P.Paths.resolve().artifacts == (P.repo_root() / "artifacts").resolve()


def test_drive_is_preferred_when_mounted(monkeypatch):
    monkeypatch.delenv("SHENG_ARTIFACTS", raising=False)
    monkeypatch.setattr(P, "_drive_available", lambda: True)
    assert P.Paths.resolve().artifacts == P.DRIVE_DEFAULT.resolve()


def test_ensure_creates_every_dir(paths):
    for d in (
        paths.artifacts,
        paths.data_dir,
        paths.wav_dir,
        paths.manifest_dir,
        paths.config_dir,
        paths.checkpoint_root,
        paths.log_root,
        paths.eval_root,
        paths.export_root,
        paths.progress_dir,
        paths.pretrained_parent,
    ):
        assert d.is_dir(), d


def test_voxcpm_repo_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXCPM_REPO", str(tmp_path / "VoxCPM"))
    assert P.Paths.resolve(tmp_path / "a").voxcpm_repo == tmp_path / "VoxCPM"


def test_train_script_and_logs(paths):
    assert paths.train_script.name == "train_voxcpm_finetune.py"
    assert paths.train_script.parent.name == "scripts"
    assert paths.trainer_log("r") == paths.checkpoint_dir("r") / "train.log"
    assert paths.train_log("r") == paths.log_dir("r") / "train.log"


def test_manifest_paths_are_relative_to_manifest_dir(paths):
    wav = paths.wav_dir / "abel_0001.wav"
    rel = paths.rel_to_manifest(wav)
    assert rel.endswith("wav16k/abel_0001.wav")
    assert not Path(rel).is_absolute()
    # The loader resolves `audio` against the manifest's own directory.
    assert (paths.manifest_dir / rel).resolve() == wav.resolve()
    assert "\\" not in rel


def test_constants():
    assert P.DATASET_ID == "lyle-mlengineer/kenyan-celebs"
    assert P.BASE_MODEL_ID == "openbmb/VoxCPM2"
    assert P.TARGET_SAMPLE_RATE == 16_000
    assert P.OUT_SAMPLE_RATE == 48_000
    assert P.DEFAULT_RUN == "lora_r32"
