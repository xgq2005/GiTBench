#!/usr/bin/env bash
# Vanilla GR00T N1.6 fine-tune (no HAMLET) -- baseline for HAMLET comparisons.
# Usage: DATASET_PATH=/path/to/robomme bash run_scripts/robomme/train_vanilla_n1d6.sh
#   RoboMME modality (8-D abs-joint / 2-view) is preset (robomme_config.py).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
DATASET_PATH="${DATASET_PATH:-${ROBOMME_DATASET_PATH:-$REPO_ROOT/data/robomme}}"
MODALITY_CONFIG="${MODALITY_CONFIG:-gr00t/configs/data/robomme_config.py}"  # robomme_config.py | rmbench_config.py
OUTPUT_DIR="${OUTPUT_DIR:-runs/robomme/vanilla_n1d6}"
BASE_MODEL="${BASE_MODEL:-$REPO_ROOT/models/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-30000}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

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

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG" \
    --hamlet-mode off \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --save-steps "$SAVE_STEPS"
