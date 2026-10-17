#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_ROOT="$ROOT/policy/pi05"
PYTHON_BIN="${GITBENCH_PYTHON:-/data/wangyi/conda/envs/maniskill/bin/python}"
SPLIT="${1:-test}"
if [[ $# -gt 0 ]]; then shift; fi
TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then TASKS=(bio2 bio4 bio5 ind2 ind3 ind5 house1 house3 house5); fi

export PI05_BACKEND=websocket
export PI05_HOST="${PI05_HOST:-127.0.0.1}"
export PI05_PORT="${PI05_PORT:-8000}"
export PYTHONPATH="$POLICY_ROOT/src:$POLICY_ROOT/packages/openpi-client/src:$POLICY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PYTHON_BIN" -c 'import msgpack, websockets' >/dev/null 2>&1; then
  echo "Missing WebSocket client dependencies. Run: bash $POLICY_ROOT/install_eval_deps.sh" >&2
  exit 1
fi
exec "$PYTHON_BIN" "$ROOT/scripts/eval_policy.py" \
  --policy pi05:make_policy --tasks "${TASKS[@]}" --split "$SPLIT"
