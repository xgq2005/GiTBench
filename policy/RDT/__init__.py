"""Factory exposed to GITBench as ``rdt:make_policy``.

The implementation is imported lazily so importing ``policy.RDT`` during test
discovery does not load heavyweight inference dependencies.
"""

from pathlib import Path
import sys


def make_policy(action_space):
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from gitbench.benchmark_policy import GITBenchPolicy

    return GITBenchPolicy(action_space)
