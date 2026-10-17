#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_ROOT="$ROOT/policy/pi05"
PYTHON_BIN="${PI05_EVAL_PYTHON:-/data/wangyi/conda/envs/openpi/bin/python}"
CHECKPOINT="${1:?Usage: $0 <checkpoint_dir> [split] [tasks...] }"
shift
SPLIT="test"
if [[ $# -gt 0 && "$1" == test* ]]; then
  SPLIT="$1"
  shift
fi
TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then TASKS=(bio2 bio4 bio5 ind2 ind3 ind5 house1 house3 house5); fi

export PI05_BACKEND=local
export PI05_CHECKPOINT="$CHECKPOINT"
export PI05_CONFIG="${PI05_CONFIG:-pi05_gitbench_lora}"
export PYTHONPATH="$POLICY_ROOT/src:$POLICY_ROOT/packages/openpi-client/src:$POLICY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$POLICY_ROOT/.cache/openpi}"
exec "$PYTHON_BIN" "$ROOT/scripts/eval_policy.py" \
  --policy pi05:make_policy --tasks "${TASKS[@]}" --split "$SPLIT"
