#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${GITBENCH_PYTHON:-/data/wangyi/conda/envs/maniskill/bin/python}"

# Keep the benchmark client aligned with OpenPI's original client package.
exec "$PYTHON_BIN" -m pip install -e "$ROOT/packages/openpi-client"
