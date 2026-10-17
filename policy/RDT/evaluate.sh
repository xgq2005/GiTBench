#!/usr/bin/env bash
set -euo pipefail

# Evaluate RDT on GITBench. Settings can be supplied through environment
# variables or command-line options; command-line values take precedence.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_ROOT="$ROOT/policy/RDT"
cd "$POLICY_ROOT"
RDT_PYTHON="${RDT_PYTHON:-/data/wangyi/conda/envs/rdt/bin/python}"
GITBENCH_PYTHON="${GITBENCH_PYTHON:-$RDT_PYTHON}"
MANISKILL_SITE_PACKAGES="${MANISKILL_SITE_PACKAGES:-/data/wangyi/conda/envs/maniskill/lib/python3.10/site-packages}"
CHECKPOINT="${RDT_CHECKPOINT:-}"
TASKS_TEXT="${RDT_EVAL_TASKS:-bio2 bio4 bio5 ind2 ind3 ind5 house1 house3 house5}"
EPISODES="${RDT_EVAL_EPISODES:-50}"
START_EPISODE="${RDT_EVAL_START_EPISODE:-0}"
MAX_STEPS="${RDT_EVAL_MAX_STEPS:-}"
SPLIT="${RDT_EVAL_SPLIT:-test}"
OBS_MODE="${RDT_EVAL_OBS_MODE:-rgbd}"
RENDER_MODE="${RDT_EVAL_RENDER_MODE:-rgb_array}"
SHADER="${RDT_EVAL_SHADER:-default}"
SIM_BACKEND="${RDT_EVAL_SIM_BACKEND:-auto}"
VIDEO_FPS="${RDT_EVAL_VIDEO_FPS:-30}"
OUTPUT_DIR="${RDT_EVAL_OUTPUT_DIR:-}"
VIDEO_DIR="${RDT_EVAL_VIDEO_DIR:-}"
SAVE_VIDEO="${RDT_EVAL_SAVE_VIDEO:-1}"
RESUME="${RDT_EVAL_RESUME:-1}"
STOP_ON_SUCCESS="${RDT_EVAL_STOP_ON_SUCCESS:-1}"
TERMINATE_ON_SUCCESS="${RDT_EVAL_TERMINATE_ON_SUCCESS:-0}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash evaluate.sh [options]

  --checkpoint PATH       RDT_CHECKPOINT
  --tasks "TASK ..."      RDT_EVAL_TASKS
  --episodes N             RDT_EVAL_EPISODES (default: 50 per task)
  --start-episode N       RDT_EVAL_START_EPISODE (default: 0)
  --max-steps N           RDT_EVAL_MAX_STEPS (default: task limit)
  --split NAME            RDT_EVAL_SPLIT (default: test)
  --output-dir DIR        RDT_EVAL_OUTPUT_DIR (default: timestamped eval_results/rdt)
  --video-dir DIR         RDT_EVAL_VIDEO_DIR (default: OUTPUT_DIR/videos)
  --video-fps N            RDT_EVAL_VIDEO_FPS (default: 30)
  --no-video              Disable mp4 generation
  --no-resume             Do not skip episodes already present in log.json
  --no-stop-on-success    Run each episode until max_steps
  --terminate-on-success  Terminate env.step when success is reached
  --dry-run               Print resolved settings without running
  -h, --help              Show this help
EOF
}

while (($#)); do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --tasks) TASKS_TEXT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --start-episode) START_EPISODE="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --no-video) SAVE_VIDEO=0; shift ;;
    --no-resume) RESUME=0; shift ;;
    --no-stop-on-success) STOP_ON_SUCCESS=0; shift ;;
    --terminate-on-success) TERMINATE_ON_SUCCESS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT/eval_results/rdt_$(date +%Y%m%d_%H%M%S)"
elif [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT/$OUTPUT_DIR"
fi
if [[ -z "$VIDEO_DIR" ]]; then
  VIDEO_DIR="$OUTPUT_DIR/videos"
elif [[ "$VIDEO_DIR" != /* ]]; then
  VIDEO_DIR="$ROOT/$VIDEO_DIR"
fi

read -r -a TASK_LIST <<< "$TASKS_TEXT"
if ((${#TASK_LIST[@]} == 0)); then
  echo "No tasks specified. Use --tasks or RDT_EVAL_TASKS." >&2
  exit 2
fi
if [[ -z "$CHECKPOINT" ]]; then
  while IFS= read -r candidate; do
    if [[ -s "$candidate/pytorch_model.bin" ]]; then
      CHECKPOINT="$candidate"
      break
    fi
  done < <(find "$POLICY_ROOT/checkpoints/rdt-mem-sim" -mindepth 1 -maxdepth 1 \
    -type d -name 'checkpoint-[0-9]*' -print 2>/dev/null | sort -Vr)
  if [[ -z "$CHECKPOINT" && -s "$POLICY_ROOT/checkpoints/rdt-mem-sim/pytorch_model.bin" ]]; then
    CHECKPOINT="$POLICY_ROOT/checkpoints/rdt-mem-sim"
  fi
fi
if [[ -z "$CHECKPOINT" || ! -e "$CHECKPOINT" ]]; then
  echo "RDT checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi

export MANISKILL_SITE_PACKAGES RDT_CHECKPOINT="$CHECKPOINT"
export RDT_LANG_EMBED_DIR="${RDT_LANG_EMBED_DIR:-$POLICY_ROOT/outs/gitbench_lang_embeds}"
export RDT_VISION_ENCODER="${RDT_VISION_ENCODER:-$POLICY_ROOT/google/siglip-so400m-patch14-384}"
export PYTHONPATH="$ROOT:$ROOT/policy:$POLICY_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ((DRY_RUN)); then
  echo "[dry-run] checkpoint: $RDT_CHECKPOINT"
  echo "[dry-run] tasks: ${TASK_LIST[*]}"
  echo "[dry-run] episodes per task: $EPISODES"
  echo "[dry-run] output: $OUTPUT_DIR"
  echo "[dry-run] video: $([[ "$SAVE_VIDEO" == 1 ]] && echo enabled || echo disabled)"
  exit 0
fi

mkdir -p "$OUTPUT_DIR" "$VIDEO_DIR"
"$RDT_PYTHON" -m scripts.encode_gitbench_eval_lang \
  --manifest="$ROOT/task_config/test_manifest.json" \
  --save_dir="$RDT_LANG_EMBED_DIR" \
  --config_path=configs/gitbench.yaml \
  --model_path="${TEXT_ENCODER_NAME:-$POLICY_ROOT/google/t5-v1_1-xxl}" \
  --split="$SPLIT" \
  --gpu="${ENCODE_GPU:-0}" \
  --offload_dir="${T5_OFFLOAD_DIR:-$POLICY_ROOT/outs/t5_offload_gitbench}" \
  --tasks "${TASK_LIST[@]}"

EVAL_CMD=("$GITBENCH_PYTHON" "$ROOT/scripts/eval_policy.py" --policy rdt:make_policy
  --tasks "${TASK_LIST[@]}" --split "$SPLIT" --episodes "$EPISODES"
  --start-episode "$START_EPISODE" --obs-mode "$OBS_MODE"
  --render-mode "$RENDER_MODE" --shader "$SHADER" --sim-backend "$SIM_BACKEND"
  --out "$OUTPUT_DIR/log.json" --summary-out "$OUTPUT_DIR/summary.json"
  --video-dir "$VIDEO_DIR" --video-fps "$VIDEO_FPS")
[[ -n "$MAX_STEPS" ]] && EVAL_CMD+=(--max-steps "$MAX_STEPS")
[[ "$SAVE_VIDEO" == 1 ]] && EVAL_CMD+=(--save-video)
[[ "$RESUME" == 1 ]] && EVAL_CMD+=(--resume) || EVAL_CMD+=(--no-resume)
[[ "$STOP_ON_SUCCESS" == 1 ]] && EVAL_CMD+=(--stop-on-success) || EVAL_CMD+=(--no-stop-on-success)
[[ "$TERMINATE_ON_SUCCESS" == 1 ]] && EVAL_CMD+=(--terminate-on-success) || EVAL_CMD+=(--no-terminate-on-success)
exec "${EVAL_CMD[@]}"
