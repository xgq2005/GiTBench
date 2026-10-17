#!/usr/bin/env bash
set -euo pipefail
POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$POLICY_ROOT"
RDT_PYTHON="${RDT_PYTHON:-/data/wangyi/conda/envs/rdt/bin/python}"
exec "$RDT_PYTHON" -m scripts.compute_gitbench_stat \
  --lerobot_root="${LEROBOT_ROOT:-/data/wangyi/xgq/GITBenchv4/data/GiTBench_train}" \
  --save_path=configs/dataset_stat.json
