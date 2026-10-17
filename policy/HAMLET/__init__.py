"""GITBench entry point for the HAMLET/GR00T policy.

The model implementation is kept in this directory so it can be evaluated from
the benchmark checkout without installing HAMLET as a second top-level package.
"""

from __future__ import annotations

from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .gitbench_policy import make_policy

__all__ = ["make_policy"]
