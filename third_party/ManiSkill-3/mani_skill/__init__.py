from pathlib import Path

from .utils.logging_utils import logger

__version__ = "3.0.0b22"

# ---------------------------------------------------------------------------- #
# Setup paths
# ---------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).parent.resolve()
PACKAGE_ASSET_DIR = PACKAGE_DIR / "assets"
# Non-package data
LOCAL_DATA_DIR = PACKAGE_DIR.parent / "gitbench_maniskill_data"
ASSET_DIR = LOCAL_DATA_DIR / "data"


def format_path(p: str):
    return p.format(
        PACKAGE_DIR=PACKAGE_DIR,
        PACKAGE_ASSET_DIR=PACKAGE_ASSET_DIR,
        ASSET_DIR=ASSET_DIR,
    )


# ---------------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------------- #
def get_commit_info(show_modified_files=False, show_untracked_files=False):
    """Get git commit information."""
    return None


from .envs import *
