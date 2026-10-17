#!/usr/bin/env bash
set -euo pipefail

POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$POLICY_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29513}"
export ENCODE_GPU="${ENCODE_GPU:-0}"
export RDT_PYTHON="${RDT_PYTHON:-/data/wangyi/conda/envs/rdt/bin/python}"
export TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-$POLICY_ROOT/google/t5-v1_1-xxl}"
export VISION_ENCODER_NAME="${VISION_ENCODER_NAME:-$POLICY_ROOT/google/siglip-so400m-patch14-384}"
export PRETRAINED_RDT="${PRETRAINED_RDT:-$POLICY_ROOT/checkpoints/rdt-1b}"
export LEROBOT_ROOT="${LEROBOT_ROOT:-$POLICY_ROOT/../../data/GiTBench_train}"
export LANG_EMBED_DIR="${LANG_EMBED_DIR:-$POLICY_ROOT/outs/gitbench_lang_embeds}"
export T5_OFFLOAD_DIR="${T5_OFFLOAD_DIR:-$POLICY_ROOT/outs/t5_offload_gitbench}"
export OUTPUT_DIR="${OUTPUT_DIR:-$POLICY_ROOT/checkpoints/rdt-gitbench}"
SWANLAB_ENABLED="${SWANLAB_ENABLED:-false}"
if [[ "$SWANLAB_ENABLED" == "true" || "$SWANLAB_ENABLED" == "1" ]]; then
  # Explicit enable wins even if the parent shell exported REPORT_TO=none.
  export REPORT_TO="swanlab"
else
  export REPORT_TO="${REPORT_TO:-none}"
fi

mkdir -p "$OUTPUT_DIR" "$LANG_EMBED_DIR" "$T5_OFFLOAD_DIR"

"$RDT_PYTHON" -m scripts.encode_lerobot_lang \
  --lerobot_root="$LEROBOT_ROOT" --save_dir="$LANG_EMBED_DIR" \
  --config_path=configs/gitbench.yaml --model_path="$TEXT_ENCODER_NAME" \
  --gpu="$ENCODE_GPU" --offload_dir="$T5_OFFLOAD_DIR"

LAUNCH_ARGS=(--num_processes="$NUM_PROCESSES" --main_process_port="$MAIN_PROCESS_PORT")
if (( NUM_PROCESSES > 1 )); then LAUNCH_ARGS=(--multi_gpu "${LAUNCH_ARGS[@]}"); fi
"$RDT_PYTHON" -m accelerate.commands.launch "${LAUNCH_ARGS[@]}" main.py \
  --config_path=configs/gitbench.yaml --deepspeed=configs/zero2.json \
  --pretrained_model_name_or_path="$PRETRAINED_RDT" \
  --pretrained_text_encoder_name_or_path="$TEXT_ENCODER_NAME" \
  --pretrained_vision_encoder_name_or_path="$VISION_ENCODER_NAME" \
  --output_dir="$OUTPUT_DIR" --train_batch_size="$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps="$GRAD_ACCUM" \
  --sample_batch_size=2 --num_sample_batches=1 --max_train_steps="${MAX_TRAIN_STEPS:-60000}" \
  --checkpointing_period="${CHECKPOINTING_PERIOD:-5000}" --sample_period="${SAMPLE_PERIOD:-1000}" \
  --checkpoints_total_limit="${CHECKPOINTS_TOTAL_LIMIT:-20}" --lr_scheduler=constant \
  --learning_rate="${LEARNING_RATE:-1e-4}" --mixed_precision=bf16 \
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS:-2}" --image_aug \
  --dataset_type=finetune --state_noise_snr=40 \
  --lerobot_root="$LEROBOT_ROOT" --lang_embed_dir="$LANG_EMBED_DIR" \
  --precomp_lang_embed --report_to="${REPORT_TO:-none}" \
  --swanlab_project="${SWANLAB_PROJECT:-GITBench}" \
  --swanlab_experiment_name="${SWANLAB_EXPERIMENT_NAME:-rdt_gitbench}" \
  ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint="$RESUME_FROM_CHECKPOINT"}
