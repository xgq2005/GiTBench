"""GITBench factories for direct and WebSocket OpenPI pi05 inference."""

from __future__ import annotations

from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT / "src", _ROOT / "packages" / "openpi-client" / "src", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

def make_policy(action_space):
    """Default to WebSocket mode; use PI05_BACKEND=local for in-process inference."""
    # Import lazily to keep this package import-safe during unittest discovery.
    from gitbench.benchmark_policy import make_local_policy, make_websocket_policy

    if __import__("os").environ.get("PI05_BACKEND", "websocket") == "local":
        return make_local_policy(action_space)
    return make_websocket_policy(action_space)


__all__ = ["make_policy"]
