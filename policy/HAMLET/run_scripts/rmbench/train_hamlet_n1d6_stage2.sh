#!/usr/bin/env bash
# Stage 2: HAMLET fine-tune loading Stage-1 TCL-pretrained moment tokens
# Usage: bash run_scripts/rmbench/train_hamlet_n1d6_stage2.sh
#   Point STAGE1_CKPT to the Stage-1 checkpoint directory.
set -euo pipefail

# Activate conda env
CONDA_BASE="${CONDA_BASE:-}"
CONDA_ENV="${CONDA_ENV:-hamlet}"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
# Ensure ffmpeg for torchcodec video decoding (run once: conda install -c conda-forge ffmpeg=6.1.1)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
DATASET_PATH="${DATASET_PATH:-${RMBENCH_DATASET_PATH:-$REPO_ROOT/data/rmbench}}"
MODALITY_CONFIG="${MODALITY_CONFIG:-gr00t/configs/data/rmbench_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/rmbench/hamlet_n1d6_stage2}"
BASE_MODEL="${BASE_MODEL:-models/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"           # per_device=2 → 4gpu×2accum=16 effective batch
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-30000}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

# HAMLET memory options
K="${K:-4}"
MEMORY_STRIDE="${MEMORY_STRIDE:-16}"
N_MOMENT_TOKENS="${N_MOMENT_TOKENS:-4}"
MEM_COND_TYPE="${MEM_COND_TYPE:-cross_attn}"
MEMORY_TYPE="${MEMORY_TYPE:-moment_token}"

# Stage-1 TCL checkpoint path (REQUIRED for Stage 2)
STAGE1_CKPT="${STAGE1_CKPT:-runs/rmbench/hamlet_n1d6_stage1_tcl/checkpoint-30000}"
FREEZE_MOMENT_TOKENS="${FREEZE_MOMENT_TOKENS:-0}"   # 0 = unfreeze (best perf), 1 = freeze (cost-saving)

if [ ! -d "$STAGE1_CKPT" ] && [ ! -f "$STAGE1_CKPT" ]; then
    echo "[ERROR] Stage-1 checkpoint not found: $STAGE1_CKPT"
    echo "  Set STAGE1_CKPT to your Stage-1 (TCL) checkpoint path."
    exit 1
fi

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
MOMENT_ARGS+=(--load-moment-tokens-from "$STAGE1_CKPT")

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
    --hamlet-mode finetune \
    --n-moment-tokens "$N_MOMENT_TOKENS" \
    --memory-window "$K" \
    --memory-stride "$MEMORY_STRIDE" \
    --memory-num-layers 2 \
    --mem-cond-type "$MEM_COND_TYPE" \
    --memory-type "$MEMORY_TYPE" \
    "${MOMENT_ARGS[@]}"
