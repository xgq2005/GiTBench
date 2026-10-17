#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_PYTHON="${OPENPI_PYTHON:-/data/wangyi/conda/envs/openpi/bin/python}"
# Use four GPUs by default; callers can override this for another allocation.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODE="${1:-lora}"
RUN_NAME="${2:-pi05}"
CONFIG="pi05_gitbench_lora"
if [[ "$MODE" == "full" ]]; then
  CONFIG="pi05_gitbench"
elif [[ "$MODE" != "lora" ]]; then
  echo "Usage: $0 [lora|full] [run_name]" >&2
  exit 2
fi
shift $(( $# >= 2 ? 2 : $# ))

export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ROOT/.cache/openpi}"
# Do not reserve almost the entire GPU before the first compiled step. This is
# safer on 32-GiB cards and leaves room for CUDA/cuDNN workspaces.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
# Avoid known cuDNN autotune plans that can crash on RTX 5090 with the
# environment's JAX/cuDNN combination. This is a workaround for execution
# failures, not a replacement for upgrading to a Blackwell-compatible JAX.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
SWANLAB_ENABLED="${SWANLAB_ENABLED:-false}"
SWANLAB_ARGS=()
if [[ "$SWANLAB_ENABLED" == "true" || "$SWANLAB_ENABLED" == "1" ]]; then
  SWANLAB_ARGS+=(--swanlab-enabled)
else
  SWANLAB_ARGS+=(--no-swanlab-enabled)
fi
cd "$ROOT"
"$OPENPI_PYTHON" scripts/compute_norm_stats.py --config-name "$CONFIG"
exec "$OPENPI_PYTHON" scripts/train.py "$CONFIG" \
  --project-name="${SWANLAB_PROJECT:-GITBench}" \
  --exp-name="$RUN_NAME" \
  "${SWANLAB_ARGS[@]}" \
  "$@"
