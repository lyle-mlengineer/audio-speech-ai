from __future__ import annotations

import pytest

from sheng_qwentts.env import check, suggest_profile


@pytest.mark.parametrize(
    "gpu,gb,expected",
    [
        ("Tesla T4", 15.6, "t4"),
        ("NVIDIA L4", 22.5, "l4"),
        ("NVIDIA A100-SXM4-40GB", 39.5, "a100"),
        ("Quadro M2000", 4.0, "cpu"),
        (None, None, "cpu"),
    ],
)
def test_profile_suggestion(gpu, gb, expected):
    assert suggest_profile({"cuda": bool(gpu), "gpu": gpu, "gpu_gb": gb}) == expected


def test_missing_qwen_tts_is_reported_not_raised():
    info = check(
        {
            "torch": "2.6.0",
            "qwen_tts": None,
            "transformers": None,
            "soundfile": "0.12.1",
            "pyarrow": "14.0.1",
            "cuda": True,
            "gpu": "Tesla T4",
            "gpu_gb": 15.6,
        }
    )
    assert any("qwen-tts" in p for p in info["problems"])
    assert info["suggested_profile"] == "t4"


def test_transformers_drift_is_reported():
    info = check(
        {
            "torch": "2.6.0",
            "qwen_tts": "0.1.1",
            "transformers": "4.58.0",
            "soundfile": "0.12.1",
            "pyarrow": "14.0.1",
            "cuda": True,
            "gpu": "NVIDIA L4",
            "gpu_gb": 22.5,
        }
    )
    assert any("transformers" in p for p in info["problems"])


def test_no_cuda_blocks_training_but_not_the_rest():
    info = check(
        {
            "torch": "2.6.0",
            "qwen_tts": "0.1.1",
            "transformers": "4.57.3",
            "soundfile": "0.12.1",
            "pyarrow": "14.0.1",
            "cuda": False,
            "gpu": None,
            "gpu_gb": None,
        }
    )
    assert any("CUDA" in p for p in info["problems"])
    assert info["suggested_profile"] == "cpu"
