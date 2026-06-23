#!/usr/bin/env bash
set -euo pipefail

# Local first-pass mini training. This uses the first-class Oxi mini preset:
# 128d / 6L / 4H, policy-focused losses, and post-training whitening.
#
# Usage:
#   oxi/scripts/train_mini_local.sh [DATA_PATH] [LOG_DIR]
#
# Examples:
#   oxi/scripts/train_mini_local.sh ./data ./mini_local
#   TIMEOUT=3600 MAX_SAMPLES=200000 oxi/scripts/train_mini_local.sh ./data ./mini_smoke

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_PATH="${1:-$ROOT/../data}"
LOG_DIR="${2:-$ROOT/mini_local}"
TIMEOUT="${TIMEOUT:-3600}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

cd "$ROOT"

args=(
  run --release --features "train,backend-tch" -- train
  --model-size mini
  --data-path "$DATA_PATH"
  --log-dir "$LOG_DIR"
  --physical-batch-size 0
  --disable-tui
)

if [[ "$TIMEOUT" != "0" ]]; then
  args+=(--timeout "$TIMEOUT")
fi

if [[ "$MAX_SAMPLES" != "0" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi

exec cargo "${args[@]}"
