#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_PYTHON="${OPENPI_PYTHON:-/data/wangyi/conda/envs/openpi/bin/python}"
CHECKPOINT="${1:?Usage: $0 <checkpoint_dir> [port] [config]}"
PORT="${2:-8000}"
CONFIG="${3:-pi05_gitbench_lora}"

export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ROOT/.cache/openpi}"
cd "$ROOT"
exec "$OPENPI_PYTHON" scripts/serve_policy.py policy:checkpoint \
  --policy.config="$CONFIG" \
  --policy.dir="$CHECKPOINT" \
  --port="$PORT"
