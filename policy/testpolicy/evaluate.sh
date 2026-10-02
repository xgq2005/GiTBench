#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
else
  echo "No Python interpreter found. Activate the gitbench environment or set PYTHON_BIN." >&2
  exit 127
fi

exec "$python_bin" "$ROOT/scripts/evaluate_policy_config.py" \
  --config "$ROOT/policy/testpolicy/eval_config.yaml" \
  "$@"
