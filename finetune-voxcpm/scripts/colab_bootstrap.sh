#!/usr/bin/env bash
# Install this pipeline on a fresh Colab VM (or any Linux box) in one shot.
#
# The heavy ML stack (torch/torchaudio) is NOT installed here: Colab already
# ships a CUDA-matched build and replacing it is the classic way to break a
# session. `sheng-voxcpm env` installs OpenBMB/VoxCPM with --no-deps plus a
# curated dependency list instead.
#
# Usage:
#   bash scripts/colab_bootstrap.sh                  # install the code only
#   bash scripts/colab_bootstrap.sh --with-env       # ...and clone/install VoxCPM
#   bash scripts/colab_bootstrap.sh --artifacts /content/drive/MyDrive/voxcpm2-sheng
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_ENV=0
ARTIFACTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-env) WITH_ENV=1; shift ;;
    --artifacts) ARTIFACTS="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "== python =="
python -VV

echo "== installing $REPO_DIR (editable, light deps only) =="
python -m pip install -q -e "$REPO_DIR"
python -m pip install -q "pyyaml>=6.0" "tqdm>=4.66"

ARGS=()
if [[ -n "$ARTIFACTS" ]]; then
  ARGS+=(--artifacts "$ARTIFACTS")
fi

if [[ $WITH_ENV -eq 1 ]]; then
  echo "== cloning + installing OpenBMB/VoxCPM =="
  sheng-voxcpm "${ARGS[@]}" env
fi

echo "== doctor =="
sheng-voxcpm "${ARGS[@]}" doctor || true

cat <<'EOF'

Next:
  sheng-voxcpm all --resume --max-hours 8

After a disconnect, re-run exactly that command: finished stages are skipped and
training resumes from <artifacts>/checkpoints/<run>/latest.
EOF
