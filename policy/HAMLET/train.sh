#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:-$ROOT/../../data/GiTBench_train}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$ROOT/gr00t/configs/data/gitbench_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/gitbench/hamlet_n1d6}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/GR00T-N1.6-3B}"
HAMLET_PYTHON="${HAMLET_PYTHON:-/data/wangyi/conda/envs/hamlet/bin/python}"
SWANLAB_PROJECT_NAME="${SWANLAB_PROJECT:-GITBench}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-hamlet_gitbench}"
SWANLAB_ENABLED="${SWANLAB_ENABLED:-false}"
SWANLAB_ARGS=()
if [[ "$SWANLAB_ENABLED" == "true" || "$SWANLAB_ENABLED" == "1" ]]; then
  SWANLAB_ARGS+=(--use-swanlab)
else
  SWANLAB_ARGS+=(--no-use-swanlab)
fi
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
# 2 samples/device × 4 GPUs × 8 accumulation = effective global batch 64.
# The smaller microbatch plus activation checkpointing is the conservative
# default for the 3B HAMLET model on 32-GiB GPUs.
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"

OPTIONAL_ARGS=(
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
  --shard-size "$SHARD_SIZE"
  --episode-sampling-rate "$EPISODE_SAMPLING_RATE"
  --num-shards-per-epoch "$NUM_SHARDS_PER_EPOCH"
)
if [[ "${SKIP_WEIGHT_LOADING:-false}" == "true" || "${SKIP_WEIGHT_LOADING:-false}" == "1" ]]; then
  OPTIONAL_ARGS+=(--skip-weight-loading)
fi

for required_path in "$DATASET_PATH" "$BASE_MODEL" "$MODALITY_CONFIG" "$HAMLET_PYTHON"; do
  if [[ ! -e "$required_path" ]]; then
    echo "[ERROR] Required path does not exist: $required_path" >&2
    exit 1
  fi
done

cd "$ROOT"
# SwanLab 0.9.x treats SWANLAB_PROJECT as a JSON-encoded settings object when
# reading the environment.  The CLI below receives the plain project name, so
# remove the conflicting environment variable before Python imports SwanLab.
# SwanLab also reserves SWANLAB_EXPERIMENT for a JSON-encoded settings
# object.  A plain experiment name in that variable makes pydantic fail while
# importing SwanLab; the CLI argument below is the supported way to set it.
unset SWANLAB_PROJECT SWANLAB_EXPERIMENT
exec "$HAMLET_PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  -m gr00t.experiment.launch_finetune \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$OUTPUT_DIR" \
  --swanlab-project "$SWANLAB_PROJECT_NAME" \
  "${SWANLAB_ARGS[@]}" \
  --experiment-name "$SWANLAB_EXPERIMENT_NAME" \
  --max-steps "$MAX_STEPS" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --save-steps "$SAVE_STEPS" \
  "${OPTIONAL_ARGS[@]}" \
  --video-backend "${VIDEO_BACKEND:-pyav}" \
  --hamlet-mode finetune \
  --n-moment-tokens "${N_MOMENT_TOKENS:-4}" \
  --memory-window "${MEMORY_WINDOW:-8}" \
  --memory-stride "${MEMORY_STRIDE:-16}" \
  --memory-num-layers 2 \
  --mem-cond-type "${MEM_COND_TYPE:-cross_attn}" \
  --memory-type "${MEMORY_TYPE:-moment_token}" \
  --no-freeze-moment-tokens
