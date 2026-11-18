"""Artifacts-root resolution and the derived path layout."""

from __future__ import annotations

from pathlib import Path

from sheng_chatterbox import paths as P


def test_explicit_artifacts_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(P.ENV_VAR, str(tmp_path / "from-env"))
    assert P.resolve_artifacts(tmp_path / "explicit") == (tmp_path / "explicit").resolve()


def test_env_var_when_no_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(P.ENV_VAR, str(tmp_path / "from-env"))
    assert P.resolve_artifacts() == (tmp_path / "from-env").resolve()


def test_repo_artifacts_without_env_or_drive(monkeypatch):
    monkeypatch.delenv(P.ENV_VAR, raising=False)
    monkeypatch.setattr(P, "colab_drive_mounted", lambda: False)
    assert P.resolve_artifacts() == (P.REPO_ROOT / "artifacts").resolve()


def test_drive_is_preferred_when_mounted(monkeypatch):
    monkeypatch.delenv(P.ENV_VAR, raising=False)
    monkeypatch.setattr(P, "colab_drive_mounted", lambda: True)
    assert P.resolve_artifacts() == P.DRIVE_DEFAULT.resolve()


def test_ensure_creates_every_dir(paths):
    for d in (
        paths.artifacts,
        paths.data_dir,
        paths.wav_dir,
        paths.manifest_dir,
        paths.config_dir,
        paths.checkpoint_root,
        paths.log_root,
        paths.cache_dir,
        paths.progress_dir,
        paths.pretrained_dir,
    ):
        assert d.is_dir(), d


def test_log_paths(paths):
    assert paths.train_log("r") == paths.log_dir("r") / "train.log"
    assert paths.console_log("r") == paths.log_dir("r") / "console.log"
    assert paths.feature_dir("r").parent == paths.cache_dir


def test_manifest_paths_are_relative_to_manifest_dir(paths):
    wav = paths.wav_dir / "abel_0001.wav"
    rel = paths.rel_to_manifest(wav)
    assert rel.endswith("wav16k/abel_0001.wav")
    assert not Path(rel).is_absolute()
    assert (paths.manifest_dir / rel).resolve() == wav.resolve()
    assert "\\" not in rel


def test_checkpoint_layout_and_lookup(paths):
    assert paths.step_dir("r", 100).name == "step_0000100"
    assert P.latest_checkpoint(paths, "r") is None

    step = paths.step_dir("r", 100)
    step.mkdir(parents=True)
    (step / "t3_cfg.safetensors").write_bytes(b"x")
    assert P.latest_checkpoint(paths, "r") == step
    assert P.checkpoint_step(step) == 100

    latest = paths.latest_dir("r")
    latest.mkdir(parents=True)
    (latest / "t3_cfg.safetensors").write_bytes(b"x")
    assert P.latest_checkpoint(paths, "r") == latest
    assert P.checkpoint_step(latest) is None


def test_on_drive_detection():
    assert P.on_drive(Path("/content/drive/MyDrive/x"))
    assert not P.on_drive(Path("/tmp/artifacts"))


def test_constants():
    assert P.DATASET_ID == "lyle-mlengineer/kenyan-celebs"
    assert P.MODEL_REPO_ID == "ResembleAI/chatterbox"
    assert P.MTL_T3_MODEL == "t3_mtl23ls_v2.safetensors"
    assert P.SAMPLE_RATE == 16_000
    assert P.LANGUAGE_ID == "sw"
    assert P.DEFAULT_RUN == "sheng_t3"
    # the multilingual loader needs exactly these files
    assert set(P.PRETRAINED_FILES) == {
        "ve.pt",
        "t3_mtl23ls_v2.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
    }
