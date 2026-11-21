from __future__ import annotations

from sauti_sheng.env import check, report


def test_report_has_the_expected_keys(tmp_path):
    info = report(tmp_path / "artifacts", tmp_path / "scratch")
    for key in (
        "python", "platform", "torch", "torchaudio", "f5_tts", "accelerate",
        "datasets", "vocos", "soundfile", "pyarrow", "numpy", "scipy",
        "ffmpeg", "cuda", "gpu", "gpu_gb", "bf16", "artifacts_free_gb",
        "scratch_free_gb",
    ):
        assert key in info
    assert info["artifacts_free_gb"] is not None


def test_check_reports_missing_training_stack(tmp_path):
    info = check(report(tmp_path / "artifacts", tmp_path / "scratch"))
    assert isinstance(info["problems"], list)
    assert "suggested_profile" in info
    # f5-tts is deliberately not a hard dependency of the data/CLI package.
    if info["f5_tts"] is None:
        assert any("f5-tts is missing" in p for p in info["problems"])
    if not info["cuda"]:
        assert any("no CUDA device" in p for p in info["problems"])
