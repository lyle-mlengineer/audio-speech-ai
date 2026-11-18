#!/usr/bin/env bash
# One-shot Colab bootstrap for the Kenyan voice-design LoRA.
#
#   !bash scripts/colab_bootstrap.sh          # install + doctor
#   !bash scripts/colab_bootstrap.sh --all    # install + the whole pipeline
#
# Safe to re-run after a runtime disconnect: the pipeline skips finished stages
# and training resumes from <artifacts>/checkpoints/<run>/latest.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/lyle-mlengineer/audio-speech-ai.git}"
DEST="${DEST:-/content/audio-speech-ai}"
SUBDIR="voice-design"
ARTIFACTS="${ARTIFACTS:-/content/drive/MyDrive/kenya-voices}"

echo "[bootstrap] GPU:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || true

if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 "$REPO_URL" "$DEST"
fi
cd "$DEST/$SUBDIR"

# Mount Drive before anything writes artifacts, so a disconnect costs nothing.
python - <<'PY'
from pathlib import Path
try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
except Exception as exc:  # not on Colab
    print(f"[bootstrap] Drive not mounted: {exc}")
print("[bootstrap] Drive mounted:", Path("/content/drive/MyDrive").is_dir())
PY

pip install -q -e . || pip install -q -e ".[full]" || true

export KENYA_ARTIFACTS="$ARTIFACTS"
kenya-voices doctor || true

if [ "${1:-}" = "--all" ]; then
  kenya-voices all --resume --max-hours "${MAX_HOURS:-8}"
fi
