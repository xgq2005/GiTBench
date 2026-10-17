#!/usr/bin/env bash
# Stage 1: TCL (Time-Contrastive Learning) pretraining of moment tokens
# Pretrains the HAMLET moment tokens with VLM backbone frozen.
# Usage: DATASET_PATH=/path/to/rmbench bash run_scripts/rmbench/train_hamlet_n1d6_stage1_tcl.sh
set -euo pipefail

# Activate conda env (adjust path if needed)
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
OUTPUT_DIR="${OUTPUT_DIR:-runs/rmbench/hamlet_n1d6_stage1_tcl}"
BASE_MODEL="${BASE_MODEL:-models/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-30000}"   # TCL is cheaper; can converge in fewer steps
SAVE_STEPS="${SAVE_STEPS:-15000}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

# HAMLET TCL options
N_MOMENT_TOKENS="${N_MOMENT_TOKENS:-4}"
TCL_TAU="${TCL_TAU:-0.07}"

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
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --save-steps "$SAVE_STEPS" \
    --hamlet-mode tcl \
    --n-moment-tokens "$N_MOMENT_TOKENS" \
    --tcl-tau "$TCL_TAU"
