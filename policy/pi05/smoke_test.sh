#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_PYTHON="${OPENPI_PYTHON:-/data/wangyi/conda/envs/openpi/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ROOT/.cache/openpi}"
exec "$OPENPI_PYTHON" -m gitbench.validate "$@"
