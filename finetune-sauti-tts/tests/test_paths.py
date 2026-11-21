from __future__ import annotations

from pathlib import Path

from sauti_sheng.paths import (
    Paths,
    all_checkpoints,
    checkpoint_update,
    is_f5_checkpoint,
    on_drive,
    resolve_artifacts,
    resolve_scratch,
)


def test_explicit_artifacts_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("SAUTI_SHENG_ARTIFACTS", str(tmp_path / "env"))
    assert resolve_artifacts(tmp_path / "explicit") == (tmp_path / "explicit").resolve()


def test_env_var_is_second(tmp_path, monkeypatch):
    monkeypatch.setenv("SAUTI_SHENG_ARTIFACTS", str(tmp_path / "env"))
    assert resolve_artifacts() == (tmp_path / "env").resolve()


def test_scratch_defaults_under_artifacts_when_not_on_colab(tmp_path, monkeypatch):
    monkeypatch.delenv("SAUTI_SHENG_SCRATCH", raising=False)
    # /content does not exist in the test environment, so scratch falls back.
    scratch = resolve_scratch(tmp_path / "scratch")
    assert scratch == (tmp_path / "scratch").resolve()


def test_layout_is_stable(tmp_path):
    paths = Paths(tmp_path / "artifacts").ensure()
    assert paths.wav_dir == paths.data_dir / "wav24k"
    assert paths.f5_raw_arrow == paths.data_dir / "f5" / "raw.arrow"
    assert paths.pretrained_pointer("run") == paths.checkpoint_dir("run") / "pretrained_sauti.pt"
    assert paths.latest_checkpoint("run").name == "model_last.pt"
    assert paths.ref_audio.name == "ref_audio.wav"


def test_rel_is_short_inside_artifacts(tmp_path):
    paths = Paths(tmp_path / "artifacts").ensure()
    assert paths.rel(paths.wav_dir / "a.wav") == "data/wav24k/a.wav"


def test_checkpoint_update_reads_filename(tmp_path):
    assert checkpoint_update(tmp_path / "model_250.pt") == 250
    assert checkpoint_update(tmp_path / "model_last.pt") is None
    assert checkpoint_update(tmp_path / "pretrained_sauti.pt") is None


def test_all_checkpoints_sorts_and_ignores_pretrained(tmp_path):
    paths = Paths(tmp_path / "artifacts").ensure()
    ckpt = paths.checkpoint_dir("run")
    ckpt.mkdir(parents=True)
    for name in ("model_100.pt", "model_50.pt", "model_last.pt", "pretrained_sauti.pt", "notes.txt"):
        (ckpt / name).write_bytes(b"x")
    assert [p.name for p in all_checkpoints(paths, "run")] == ["model_50.pt", "model_100.pt"]


def test_is_f5_checkpoint(tmp_path):
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    assert is_f5_checkpoint(pt)
    assert not is_f5_checkpoint(tmp_path / "model.bin")
    assert not is_f5_checkpoint(tmp_path / "missing.pt")


def test_on_drive():
    assert on_drive(Path("/content/drive/MyDrive/x"))
    assert not on_drive(Path("/home/me/x"))
