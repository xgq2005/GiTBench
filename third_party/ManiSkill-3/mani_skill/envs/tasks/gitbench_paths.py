"""Project-local paths for the vendored GITBench task code."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def project_asset_path(*parts: str) -> Path:
    return project_root() / "asset" / Path(*parts)
