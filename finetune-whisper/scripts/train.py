"""End-to-end fine-tuning entrypoint for Sheng Whisper.

Usage:
    python scripts/train.py --config configs/whisper_medium_t4.yaml
    python scripts/train.py --config configs/whisper_medium_t4.yaml --smoke-test
    python scripts/train.py --config configs/whisper_medium_t4.yaml \
        --set train.learning_rate=2e-5 model.name=openai/whisper-small

The script is written to be resumable: if Colab dies mid-run, re-running with
the same config picks up the newest checkpoint in `output_dir`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Make `src/` importable when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_sheng.config import Config, save_config  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune Whisper for Sheng ASR")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a config value, repeatable",
    )
    p.add_argument("--smoke-test", action="store_true", help="Tiny run to validate plumbing")
    p.add_argument("--no-push", action="store_true", help="Skip pushing to the HF Hub")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint dir to resume from")
    p.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    p.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    return p.parse_args(argv)


def find_latest_checkpoint(output_dir: Path):
    if not output_dir.exists():
        return None
    ckpts = [d for d in output_dir.glob("checkpoint-*") if d.is_dir()]
    if not ckpts:
        return None

    def _step(d: Path) -> int:
        try:
            return int(d.name.split("-")[-1])
        except ValueError:
            return -1

    return str(max(ckpts, key=_step))


def main(argv=None) -> int:
    args = parse_args(argv)

    overrides = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()

    cfg = Config.from_yaml(args.config)
    if overrides:
        cfg.apply_overrides(overrides)

    if args.smoke_test:
        cfg.train.smoke_test_rows = cfg.train.smoke_test_rows or 8
        cfg.train.max_epochs = 1
        cfg.train.max_steps = 4
        cfg.train.eval_steps = 2
        cfg.train.save_steps = 2
        cfg.train.logging_steps = 1
        cfg.hub.push_to_hub = False
        cfg.train.gradient_accumulation_steps = 1

    if args.no_push:
        cfg.hub.push_to_hub = False
    if args.resume:
        cfg.train.resume_from_checkpoint = args.resume

    cfg.validate()

    if args.dry_run:
        import yaml

        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return 0

    output_dir = Path(cfg.train.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "run_config.yaml")

    print("=" * 72)
    print(f"  Whisper fine-tune for Sheng")
    print(f"  model        : {cfg.model.name}")
    print(f"  dataset      : {cfg.data.dataset_name}")
    print(f"  language/task: {cfg.model.language} / {cfg.model.task}")
    print(f"  output       : {output_dir}")
    print("=" * 72)

    # ---- imports that need the heavy deps ----------------------------------
    import torch
    from transformers import Seq2SeqTrainer

    from whisper_sheng.data import build_datasets
    from whisper_sheng.tracking import (
        build_trainer_callbacks,
        init_mlflow,
        log_config_params,
        log_dataset_stats,
        log_eval_report,
    )
    from whisper_sheng.trainer import (
        MLflowMetricCallback,
        build_compute_metrics,
        build_training_arguments,
        full_eval,
        load_model,
        make_data_collator,
        save_checkpoint_bundle,
    )

    if not torch.cuda.is_available():
        print("[warn] No CUDA device. This will be extremely slow on CPU.")
    else:
        print(f"[gpu]  {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ---- data --------------------------------------------------------------
    print("\n[1/5] Loading and preprocessing dataset...")
    bundle = build_datasets(cfg)
    stats = bundle.dataset_stats
    print(f"      train={stats['n_train']}  eval={stats['n_eval']}  "
          f"split={stats.get('split_strategy')}")
    if stats.get("single_speaker"):
        print("      [note] Dataset has a single speaker: eval WER measures")
        print("             memorisation-adjacent performance, not new-speaker")
        print("             generalisation. Treat it as a regression signal only.")

    # ---- model -------------------------------------------------------------
    print("\n[2/5] Loading model...")
    model = load_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      {cfg.model.name}: {n_params/1e6:.1f}M params")

    data_collator = make_data_collator(bundle.processor)
    training_args = build_training_arguments(cfg, output_dir)

    optimizer_cls = None
    if cfg.model.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
            print("      optimizer: AdamW8bit (bitsandbytes)")
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"      [warn] bitsandbytes unavailable ({exc}); using fp32 AdamW")

    # ---- tracking ----------------------------------------------------------
    print("\n[3/5] Starting MLflow run...")
    run = init_mlflow(cfg, run_name=args.run_name)
    import mlflow

    log_config_params(cfg)
    log_dataset_stats(stats)
    mlflow.log_dict(cfg.to_dict(), "config/resolved_config.json")
    mlflow.set_tags(
        {
            "model_name": cfg.model.name,
            "dataset": cfg.data.dataset_name,
            "language": cfg.model.language,
            "smoke_test": str(bool(args.smoke_test)).lower(),
        }
    )
    print(f"      run_id: {run.info.run_id}")
    print(f"      ui:     {cfg.mlflow.tracking_uri}")

    callbacks, progress = build_trainer_callbacks(
        output_dir / "trainer_progress.json", run.info.run_id
    )
    callbacks.append(MLflowMetricCallback())

    # Early stopping only makes sense with more than a couple of eval points.
    if cfg.train.max_steps <= 0 or cfg.train.max_steps > cfg.train.eval_steps * 3:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stopping_patience)
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=bundle.train,
        eval_dataset=bundle.eval,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(bundle.processor, cfg),
        callbacks=callbacks,
        processing_class=bundle.processor,
    )

    # ---- train -------------------------------------------------------------
    print("\n[4/5] Training...")
    resume = cfg.train.resume_from_checkpoint or find_latest_checkpoint(output_dir)
    if resume and Path(resume).exists():
        print(f"      resuming from {resume}")
    else:
        resume = None

    exit_code = 0
    try:
        train_result = trainer.train(resume_from_checkpoint=resume)
        mlflow.log_metrics(
            {f"train_{k}": float(v) for k, v in train_result.metrics.items()
             if isinstance(v, (int, float))}
        )
        progress.update(status="train_complete")
    except KeyboardInterrupt:
        print("\n[interrupted] Saving emergency checkpoint before exit...")
        emergency = output_dir / "interrupted"
        trainer.save_model(str(emergency))
        progress.update(status="interrupted")
        mlflow.set_tag("status", "interrupted")
        exit_code = 130
        raise SystemExit(exit_code)
    except Exception:
        traceback.print_exc()
        progress.update(status="failed", error=traceback.format_exc()[-2000:])
        mlflow.set_tag("status", "failed")
        mlflow.end_run(status="FAILED")
        return 1

    # ---- eval + save -------------------------------------------------------
    print("\n[5/5] Final evaluation and export...")
    report, refs, hyps = full_eval(model, bundle.processor, bundle.eval, cfg)
    print(f"      WER={report.wer:.4f}  CER={report.cer:.4f}  "
          f"mixedWER={report.mixed_wer:.4f}")

    metrics = {f"final_{k}": v for k, v in report.to_metrics_dict().items()}
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    log_eval_report(report, output_dir / "eval" / "eval_report.json")

    # Keep a human-readable sample of predictions; fastest way to eyeball
    # whether Sheng-specific slang is being learned.
    samples_path = output_dir / "eval" / "predictions.txt"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as fh:
        for row in report.per_example:
            fh.write(f"--- #{row['index']} WER={row['wer']:.3f} ---\n")
            fh.write(f"REF: {row['reference']}\n")
            fh.write(f"HYP: {row['hypothesis']}\n\n")
    mlflow.log_artifact(str(samples_path), artifact_path="eval")

    final_dir = save_checkpoint_bundle(model, bundle.processor, cfg, output_dir / "final")
    print(f"      saved to {final_dir}")

    # ---- push to hub -------------------------------------------------------
    if cfg.hub.push_to_hub:
        print(f"\n[hub] Pushing to {cfg.hub.repo_id} ...")
        try:
            _push_to_hub(cfg, model, bundle.processor, report, final_dir)
            mlflow.set_tag("hf_repo", cfg.hub.repo_id)
            print(f"      https://huggingface.co/{cfg.hub.repo_id}")
        except Exception as exc:
            traceback.print_exc()
            print(f"      [error] push failed: {exc}")
            mlflow.set_tag("hf_push_error", str(exc)[:250])
            exit_code = 1
    else:
        print("\n[hub] push_to_hub disabled; model stays local.")

    progress.update(status="complete", final_wer=report.wer)
    mlflow.set_tag("status", "complete")
    mlflow.end_run()
    print(f"\n[done] WER={report.wer:.4f}. Progress log: {output_dir/'trainer_progress.json'}")
    return exit_code


def _push_to_hub(cfg: Config, model, processor, report, final_dir: Path) -> None:
    """Upload weights, processor, metrics and a model card."""
    from huggingface_hub import HfApi, create_repo

    create_repo(cfg.hub.repo_id, private=cfg.hub.private, exist_ok=True)

    # `push_to_hub` on the objects writes directly to the repo.
    model.push_to_hub(
        cfg.hub.repo_id, commit_message=cfg.hub.commit_message, private=cfg.hub.private
    )
    processor.push_to_hub(
        cfg.hub.repo_id, commit_message=cfg.hub.commit_message, private=cfg.hub.private
    )

    api = HfApi()
    if cfg.hub.upload_eval_report:
        card = _build_model_card(cfg, report)
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=cfg.hub.repo_id,
            commit_message="Add Sheng model card with eval metrics",
        )
        report_path = final_dir / "eval_report.json"
        if report_path.exists():
            api.upload_file(
                path_or_fileobj=str(report_path),
                path_in_repo="eval_report.json",
                repo_id=cfg.hub.repo_id,
            )


def _build_model_card(cfg: Config, report) -> str:
    return f"""---
language:
- sw
- en
license: apache-2.0
tags:
- whisper
- automatic-speech-recognition
- sheng
- code-switching
- swahili
- kenyan
base_model: {cfg.model.name}
datasets:
- {cfg.data.dataset_name}
metrics:
- wer
- cer
---

# {cfg.hub.repo_id.split('/')[-1]}

Fine-tuned [{cfg.model.name}](https://huggingface.co/{cfg.model.name}) for **Sheng**,
the Swahili-English code-switched variety spoken in Kenya.

## Evaluation

| Metric | Value |
| --- | --- |
| WER | {report.wer:.4f} |
| CER | {report.cer:.4f} |
| WER (code-switched utterances only) | {report.mixed_wer:.4f} |
| Code-switch rate error | {report.code_switch_rate_error:.4f} |
| Eval examples | {report.n_examples} |

## Important caveats

Trained on `{cfg.data.dataset_name}`: **{report.n_examples}-example held-out split from a
42-row, single-speaker dataset**. Reported WER is a regression signal on this
speaker, *not* an estimate of generalisation to new Sheng speakers. Treat the
model as a starting point, not a production system.

## Usage

```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="{cfg.hub.repo_id}",
    generate_kwargs={{"language": "sw", "task": "transcribe"}},
)
print(asr("audio.wav")["text"])
```

Whisper has no dedicated Sheng language token, so the language is pinned to
Swahili (`sw`). English insertions are still produced because only the decoder
*overture* is constrained, not the output vocabulary.
"""


if __name__ == "__main__":
    raise SystemExit(main())
