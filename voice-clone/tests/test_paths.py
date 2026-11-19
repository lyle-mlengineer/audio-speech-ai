import sys
from pathlib import Path

from sheng_voiceclone.paths import (
    DATASET_ID,
    DRIVE_DEFAULT,
    ENV_VAR,
    REPO_ROOT,
    Paths,
    default_root,
    resolve,
)


def test_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "from-env"))
    assert default_root() == tmp_path / "from-env"
    assert resolve().root == tmp_path / "from-env"


def test_explicit_flag_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "from-env"))
    assert resolve(str(tmp_path / "explicit")).root == tmp_path / "explicit"


def test_fallback_is_the_repo_artifacts_dir(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr("sheng_voiceclone.paths.colab_drive_mounted", lambda: False)
    assert default_root() == REPO_ROOT / "artifacts"


def test_drive_is_used_when_mounted(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr("sheng_voiceclone.paths.colab_drive_mounted", lambda: True)
    assert default_root() == DRIVE_DEFAULT


def test_ensure_dirs_creates_every_derived_location(art):
    for p in (
        art.wav_dir,
        art.raw_dir,
        art.manifests,
        art.checkpoints,
        art.evals,
        art.scores,
        art.configs,
        art.logs,
        art.progress_dir,
    ):
        assert p.is_dir(), p


def test_rel_is_posix_inside_root(art):
    p = art.data_dir / "wav16k" / "u.wav"
    assert art.rel(p) == "data/wav16k/u.wav"


def test_rel_falls_back_outside_root(tmp_path):
    assert Paths(tmp_path / "art").rel(tmp_path / "elsewhere" / "x.wav").endswith("x.wav")


def test_paths_are_all_under_the_root(art):
    for p in (art.train_manifest, art.ref_audio, art.pretrained_dir, art.state_file):
        assert str(p).startswith(str(art.root))


def test_dataset_id_is_the_sheng_corpus():
    assert DATASET_ID == "lyle-mlengineer/kenyan-celebs"


def test_package_is_importable_from_the_test_shim():
    import sheng_voiceclone

    assert sheng_voiceclone.__version__
    assert Path(sheng_voiceclone.__file__).name == "__init__.py"
    assert sys.version_info >= (3, 10)
