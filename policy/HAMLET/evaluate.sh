#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANISKILL_SITE_PACKAGES="${MANISKILL_SITE_PACKAGES:-/data/wangyi/conda/envs/maniskill/lib/python3.10/site-packages}"
export MANISKILL_SITE_PACKAGES
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "${HAMLET_PYTHON:-/data/wangyi/conda/envs/hamlet/bin/python}" ]]; then
  python_bin="${HAMLET_PYTHON:-/data/wangyi/conda/envs/hamlet/bin/python}"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
else
  echo "No Python interpreter found. Activate the gitbench environment or set PYTHON_BIN." >&2
  exit 127
fi

exec "$python_bin" "$ROOT/scripts/evaluate_policy_config.py" \
  --config "$ROOT/policy/HAMLET/eval_config.yaml" \
  "$@"
