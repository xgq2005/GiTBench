"""Project-local asset checks for GITBench tasks."""

from __future__ import annotations

from pathlib import Path

from .paths import asset_root


def required_asset_paths() -> dict[str, Path]:
    """Assets referenced by the vendored task source."""
    root = asset_root()
    return {
        "chemistry metadata": root / "chemistry" / "info_raw.json",
        "bio2 wash bottle collision": root / "chemistry" / "models" / "020_wash_bottle" / "collision.ply",
        "bio2 wash bottle visual": root / "chemistry" / "models" / "020_wash_bottle" / "textured.obj",
        "bio2 beaker collision": root / "chemistry" / "models" / "016_beaker" / "collision.ply",
        "bio2 beaker visual": root / "chemistry" / "models" / "016_beaker" / "textured.obj",
        "bio4 beaker collision": root / "chemistry" / "models" / "016_beaker" / "collision.ply",
        "bio4 beaker visual": root / "chemistry" / "models" / "016_beaker" / "textured.obj",
        "bio4 tray_1 robocasa xml": (
            root
            / "scene_datasets"
            / "robocasa_dataset"
            / "assets"
            / "objects"
            / "objaverse"
            / "tray"
            / "tray_1"
            / "model.xml"
        ),
        "bio5 tube rack 04": root / "custom" / "tube04.glb",
        "bio5 tube rack 05": root / "custom" / "tube05.glb",
        "bio5 test tube collision": root / "chemistry" / "models" / "015_test_tube" / "tube_with_cube3.ply",
        "bio5 test tube visual": root / "chemistry" / "models" / "015_test_tube" / "textured.obj",
        "ind2 lidded box urdf": root / "industry" / "models" / "box_with_lid.urdf",
        "house1 jiazi": root / "custom" / "jiazi.glb",
        "house1 jiazi3": root / "custom" / "jiazi3.glb",
        "house1 plate_1 robocasa xml": (
            root
            / "scene_datasets"
            / "robocasa_dataset"
            / "assets"
            / "objects"
            / "objaverse"
            / "plate"
            / "plate_1"
            / "model.xml"
        ),
        "house3 bowl_4 robocasa xml": (
            root
            / "scene_datasets"
            / "robocasa_dataset"
            / "assets"
            / "objects"
            / "objaverse"
            / "bowl"
            / "bowl_4"
            / "model.xml"
        ),
        "house5 jiazi5": root / "custom" / "jiazi5.glb",
        "house5 guizi": root / "custom" / "guizi.glb",
        "house5 plate_7 visual": (
            root
            / "scene_datasets"
            / "robocasa_dataset"
            / "assets"
            / "objects"
            / "objaverse"
            / "plate"
            / "plate_7"
            / "visual"
            / "new.obj"
        ),
        "house5 plate_7 first collision part": (
            root
            / "scene_datasets"
            / "robocasa_dataset"
            / "assets"
            / "objects"
            / "objaverse"
            / "plate"
            / "plate_7"
            / "collision"
            / "new_coll01"
            / "output_000.obj"
        ),
    }


def patch_task_assets() -> None:
    """Backward-compatible hook: vendored scene source already uses project assets."""
    return None
