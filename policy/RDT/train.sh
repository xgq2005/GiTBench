#!/usr/bin/env bash
set -euo pipefail
POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$POLICY_ROOT/finetune_gitbench.sh" "$@"
