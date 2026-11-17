"""MLflow integration.

Colab sessions die without warning, so this does three things a bare
`report_to=["mlflow"]` does not:

  1. Uses a file-backed SQLite tracking store that can live under Drive, so
     runs survive a disconnected runtime.
  2. Flushes metrics on every logging step (no in-memory batching to lose).
  3. Mirrors lightweight run state to a JSON sidecar that `progress.md`-style
     resume logic can read without MLflow being importable.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config


def init_mlflow(cfg: Config, run_name: Optional[str] = None):
    """Configure tracking and start a run. Returns the active run object."""
    import mlflow

    tracking_uri = cfg.mlflow.tracking_uri
    # Relative sqlite paths break when cwd changes (Colab kernel restarts
    # land in a different directory), so anchor them to an absolute path.
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_path = Path(tracking_uri.replace("sqlite:///", "")).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    resolved_name = run_name or cfg.mlflow.run_name or _default_run_name(cfg)

    # `nested=False` + explicit run keeps resume semantics simple.
    run = mlflow.start_run(run_name=resolved_name)
    return run


def _default_run_name(cfg: Config) -> str:
    model_slug = cfg.model.name.split("/")[-1]
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{model_slug}-sheng-{ts}"


def log_config_params(cfg: Config) -> None:
    import mlflow

    flat = cfg.to_flat_dict()
    # MLflow caps param values at 500 chars and forbids some chars in keys.
    safe = {}
    for k, v in flat.items():
        key = k.replace(".", "_")
        val = v
        if val is None:
            val = "none"
        if isinstance(val, str) and len(val) > 500:
            val = val[:497] + "..."
        safe[key] = val
    mlflow.log_params(safe)


def log_dataset_stats(stats: Dict[str, Any]) -> None:
    import mlflow

    scalars = {}
    for k, v in stats.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            scalars[f"data_{k}"] = float(v)
        elif isinstance(v, bool):
            scalars[f"data_{k}"] = float(v)
    if scalars:
        mlflow.log_metrics(scalars)

    mlflow.set_tags(
        {
            "data_split_strategy": str(stats.get("split_strategy", "unknown")),
            "data_single_speaker": str(stats.get("single_speaker", False)).lower(),
            "data_speakers": ",".join(stats.get("speakers_train", []))[:250],
        }
    )


def log_eval_report(report, out_path: Path) -> None:
    """Persist per-example predictions as an MLflow artifact."""
    import mlflow

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    mlflow.log_artifact(str(out_path), artifact_path="eval")


class ProgressCallback:
    """Lightweight transformers callback that writes a JSON sidecar.

    We deliberately do not use the ProgressCallback name from transformers to
    avoid a confusing shadow; this one is ours.
    """

    def __init__(self, state_path: Path, run_id: Optional[str] = None):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Any] = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mlflow_run_id": run_id,
            "last_step": 0,
            "last_loss": None,
            "eval_history": [],
        }

    def update(self, **kwargs) -> None:
        self.state.update(kwargs)
        self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.state_path)


def build_trainer_callbacks(trainer_state_path: Path, run_id: Optional[str]):
    """Return (callbacks_list, progress_tracker)."""
    from transformers import TrainerCallback

    progress = ProgressCallback(trainer_state_path, run_id=run_id)

    class _ProgressTrainerCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            progress.update(
                last_step=int(state.global_step),
                last_loss=logs.get("loss"),
                epoch=logs.get("epoch"),
            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            metrics = metrics or {}
            progress.state.setdefault("eval_history", []).append(
                {
                    "step": int(state.global_step),
                    "wer": metrics.get("eval_wer"),
                    "loss": metrics.get("eval_loss"),
                }
            )
            progress.update()

        def on_save(self, args, state, control, **kwargs):
            progress.update(last_checkpoint=f"checkpoint-{state.global_step}")

    return [_ProgressTrainerCallback()], progress
