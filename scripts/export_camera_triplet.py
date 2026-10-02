#!/usr/bin/env python3
"""Export the three model-input RGB cameras directly from a GITBench renderer.

This utility starts a real ``GITBenchEnv``, calls ``reset()``, and writes the
three returned camera streams in model order: left | front | right. It never
opens an HDF5 dataset or decodes a recorded frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

# Allow direct execution from any working directory without installing the
# local benchmark package first.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gitbench.env import GITBenchEnv


def _to_numpy(value: Any) -> np.ndarray:
    for method_name in ("detach", "cpu", "numpy"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    return np.asarray(value)


def _camera_rgb(sensor_data: dict[str, Any], name: str) -> np.ndarray:
    camera = sensor_data.get(name)
    if not isinstance(camera, dict) or "rgb" not in camera:
        raise KeyError(f"Missing RGB stream {name!r}; available={list(sensor_data)}")
    image = _to_numpy(camera["rgb"])
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected RGB shape for {name!r}: {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _triplet_from_observation(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor_data = obs.get("sensor_data")
    if not isinstance(sensor_data, dict):
        raise ValueError("Renderer observation has no sensor_data mapping")
    # These are the exact streams consumed by the policy adapters.
    return (
        _camera_rgb(sensor_data, "piper_x-0-hand_camera"),
        _camera_rgb(sensor_data, "base_camera"),
        _camera_rgb(sensor_data, "piper_x-1-hand_camera"),
    )


def _save_triplet(images: tuple[np.ndarray, np.ndarray, np.ndarray], output: Path) -> tuple[int, int]:
    pil_images = [Image.fromarray(image, mode="RGB") for image in images]
    canvas_height = max(image.height for image in pil_images)
    canvas_width = sum(image.width for image in pil_images)
    # Different renderer cameras intentionally have different resolutions;
    # preserve each source aspect ratio and center wrist views vertically.
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(32, 32, 32))
    x_offset = 0
    for image in pil_images:
        y_offset = (canvas_height - image.height) // 2
        canvas.paste(image, (x_offset, y_offset))
        x_offset += image.width
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return canvas.size


def render_camera_triplet(
    task: str,
    episode_id: int,
    split: str,
    output: Path,
    *,
    shader: str = "default",
    sim_backend: str = "auto",
) -> dict[str, Any]:
    """Render one reset observation and save its three-camera visualization."""
    env = GITBenchEnv(
        task,
        obs_mode="rgbd",
        render_mode="rgb_array",
        shader=shader,
        sim_backend=sim_backend,
        max_episode_steps=1,
    )
    try:
        obs, info = env.reset(episode_id=episode_id, split=split)
        images = _triplet_from_observation(obs)
        size = _save_triplet(images, output)
        return {
            "task": task,
            "episode_id": episode_id,
            "split": split,
            "camera_order": ["left", "front", "right"],
            "camera_shapes": {
                "left": list(images[0].shape),
                "front": list(images[1].shape),
                "right": list(images[2].shape),
            },
            "triplet_size": list(size),
            "output": str(output),
            "renderer": "GITBenchEnv.reset",
            "info": info,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="bio2", help="GITBench task key")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("eval_results/rendered_camera_triplet.png"))
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    args = parser.parse_args()
    result = render_camera_triplet(
        args.task,
        args.episode_id,
        args.split,
        args.output,
        shader=args.shader,
        sim_backend=args.sim_backend,
    )
    print(
        f"wrote {result['output']} from renderer "
        f"(order={'|'.join(result['camera_order'])}, "
        f"shapes={result['camera_shapes']}, size={result['triplet_size']})"
    )


if __name__ == "__main__":
    main()
