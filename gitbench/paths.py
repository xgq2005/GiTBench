"""Path helpers for the self-contained gitbench project."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_MANISKILL_ROOT = repo_root() / "third_party" / "ManiSkill-3"


def asset_root() -> Path:
    return repo_root() / "asset"


def maniskill_root() -> Path:
    return DEFAULT_MANISKILL_ROOT


def ensure_maniskill_on_path() -> Path:
    """Prepend the project-local ManiSkill checkout used by the benchmark."""
    root = maniskill_root().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"ManiSkill root does not exist: {root}. "
            "Expected the vendored checkout under this gitbench project."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Keep the active interpreter's own site-packages (and PyTorch build) in
    # precedence when Gym/ManiSkill is installed in a separate environment.
    extra_site = os.environ.get("MANISKILL_SITE_PACKAGES")
    if extra_site:
        extra_path = Path(extra_site).resolve()
        if extra_path.is_dir() and str(extra_path) not in sys.path:
            sys.path.append(str(extra_path))

    # Matplotlib may be imported by ManiSkill; keep its cache project-local.
    mpl_cache = repo_root() / ".cache" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    return root
