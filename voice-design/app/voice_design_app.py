"""Minimal voice-design playground: type a description, hear a Kenyan voice.

    python app/voice_design_app.py --lora artifacts/exports/design_r32/step_0001200

Needs the same environment as training (torch + the VoxCPM repo) and a GPU. The
adapter directory must contain ``lora_weights.safetensors`` *and* ``lora_config.json``
- ``kenya-voices export`` produces exactly that layout.

The model is loaded once at start-up; every request only runs ``generate``.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kenya_voices.infer import _lora_config  # noqa: E402
from kenya_voices.paths import DEFAULT_RUN, Paths  # noqa: E402

DEFAULT_CAPTIONS = [
    "A Kenyan female voice in her thirties, mid-pitched, steady conversational pace, speaking Swahili",
    "A Kenyan male voice in his thirties, low-pitched, measured, unhurried pace, speaking Swahili",
    "A young Kenyan woman in her early twenties, bright, higher-pitched, brisk city pace, Swahili with some English words",
    "A Kenyan man from Nairobi in his forties, mid-pitched, steady conversational pace, speaking Swahili",
    "A Kenyan male voice in his twenties, low-pitched, brisk city pace, Swahili mixed with English (Sheng-style code-switching)",
    "A Kenyan male narrator in his fifties or older, low-pitched, measured, unhurried pace, natural Swahili pronunciation",
]


def load_captions(paths: Paths, run: str) -> list[str]:
    """Captions the adapter was actually trained on, if an export exists."""
    bundled = sorted(glob.glob(str(paths.export_root / run / "step_*" / "design_prompts.json")))
    if bundled:
        try:
            data = json.loads(Path(bundled[-1]).read_text(encoding="utf-8"))
            found = [c for c in data.get("captions", []) if c]
            if found:
                return found
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_CAPTIONS


def build_model(paths: Paths, lora_dir: Path | None, device: str = "cuda"):
    import torch

    from voxcpm.core import VoxCPM

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device - this app needs a GPU (Colab: Runtime > GPU)")

    kwargs = {}
    if lora_dir is not None:
        cfg_file = Path(lora_dir) / "lora_config.json"
        if cfg_file.exists():
            kwargs["lora_config"] = _lora_config(paths, json.loads(cfg_file.read_text(encoding="utf-8")))
        kwargs["lora_weights_path"] = str(lora_dir)
        print(f"[app] LoRA: {lora_dir}")
    else:
        print("[app] no --lora given: base VoxCPM2 voice design only")

    model = VoxCPM.from_pretrained(hf_model_id=str(paths.model_dir), load_denoiser=False, **kwargs)
    return model, int(model.tts_model.sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kenyan voice design playground")
    parser.add_argument("--lora", default=None, help="exported adapter dir (contains lora_config.json)")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="public Gradio link")
    args = parser.parse_args()

    import gradio as gr

    paths = Paths.resolve(args.artifacts).ensure()
    model, sample_rate = build_model(paths, Path(args.lora) if args.lora else None, args.device)
    captions = load_captions(paths, args.run)

    def synthesize(caption: str, text: str, seed: int, cfg_value: float):
        if not (caption or "").strip():
            raise gr.Error("Describe the voice first (gender, age, pitch, pace, language mix).")
        if not (text or "").strip():
            raise gr.Error("Add the text the voice should say.")
        audio = model.generate(
            text=f"({caption.strip().rstrip('.')}){text.strip()}",
            cfg_value=float(cfg_value),
            inference_timesteps=10,
            max_len=1000,
            normalize=False,
            denoise=False,
            retry_badcase=True,
            seed=int(seed),
        )
        # Gradio wants (sample_rate, numpy) for a playable clip.
        return (sample_rate, audio)

    with gr.Blocks(title="Kenyan voice design") as demo:
        gr.Markdown(
            "## Kenyan voice design\n"
            "Describe the voice, write what it should say, and listen. "
            "The caption is plain English: gender, age, pitch, pace, place, language mix."
        )
        with gr.Row():
            with gr.Column():
                preset = gr.Dropdown(captions, label="Captions the adapter was trained on", value=captions[0])
                caption = gr.Textbox(label="Voice description", value=captions[0], lines=3)
                text = gr.Textbox(
                    label="Text",
                    value="Habari yako! Karibu Kenya, leo tunaangalia mambo ya biashara.",
                    lines=3,
                )
                with gr.Accordion("Advanced", open=False):
                    seed = gr.Slider(0, 9999, value=42, step=1, label="Seed")
                    cfg_value = gr.Slider(1.0, 3.0, value=2.0, step=0.1, label="CFG (caption strength)")
                button = gr.Button("Design voice", variant="primary")
            with gr.Column():
                audio = gr.Audio(label="Result", type="numpy")

        preset.change(lambda value: value, inputs=preset, outputs=caption)
        button.click(synthesize, inputs=[caption, text, seed, cfg_value], outputs=audio)

    demo.launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
