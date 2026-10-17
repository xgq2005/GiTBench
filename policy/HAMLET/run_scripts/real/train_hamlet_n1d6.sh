#!/usr/bin/env bash
# Fine-tune HAMLET on the real LeRobot v2.1 bimanual dataset.
# Usage: DATASET_PATH=/path/to/lerobot_v21/real \
#        bash run_scripts/real/train_hamlet_n1d6.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DATASET_PATH="${DATASET_PATH:-${REAL_DATASET_PATH:-$REPO_ROOT/data/real}}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$REPO_ROOT/gr00t/configs/data/real_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/real/hamlet_n1d6}"
BASE_MODEL="${BASE_MODEL:-$REPO_ROOT/models/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-30000}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"

K="${K:-4}"
MEMORY_STRIDE="${MEMORY_STRIDE:-16}"
N_MOMENT_TOKENS="${N_MOMENT_TOKENS:-4}"
MEM_COND_TYPE="${MEM_COND_TYPE:-cross_attn}"
MEMORY_TYPE="${MEMORY_TYPE:-moment_token}"
LOAD_MOMENT_TOKENS_FROM="${LOAD_MOMENT_TOKENS_FROM:-}"
FREEZE_MOMENT_TOKENS="${FREEZE_MOMENT_TOKENS:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

for required_path in "$DATASET_PATH" "$BASE_MODEL" "$MODALITY_CONFIG"; do
    if [ ! -e "$required_path" ]; then
        echo "[ERROR] Required path does not exist: $required_path" >&2
        echo "        Override DATASET_PATH, BASE_MODEL, or MODALITY_CONFIG for this server." >&2
        exit 1
    fi
done
command -v torchrun >/dev/null 2>&1 || {
    echo "[ERROR] torchrun was not found in PATH; activate the training environment first." >&2
    exit 1
}

MOMENT_ARGS=()
if [ "$FREEZE_MOMENT_TOKENS" = "1" ]; then
    MOMENT_ARGS+=(--freeze-moment-tokens)
else
    MOMENT_ARGS+=(--no-freeze-moment-tokens)
fi
[ -z "$LOAD_MOMENT_TOKENS_FROM" ] || MOMENT_ARGS+=(--load-moment-tokens-from "$LOAD_MOMENT_TOKENS_FROM")

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG" \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --save-steps "$SAVE_STEPS" \
    --video-backend "$VIDEO_BACKEND" \
    --hamlet-mode finetune \
    --n-moment-tokens "$N_MOMENT_TOKENS" \
    --memory-window "$K" \
    --memory-stride "$MEMORY_STRIDE" \
    --memory-num-layers 2 \
    --mem-cond-type "$MEM_COND_TYPE" \
    --memory-type "$MEMORY_TYPE" \
    "${MOMENT_ARGS[@]}"
