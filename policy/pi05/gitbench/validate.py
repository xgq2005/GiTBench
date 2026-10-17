"""Validate the GITBench LeRobot source and pi05 input mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
from gitbench.transforms import GiTBenchInputs
from openpi.models import model as model_lib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/wangyi/xgq/GITBenchv4/data/GiTBench_train"),
    )
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    dataset = LeRobotDataset("gitbench_train", root=args.data_root, video_backend="pyav")
    sample = dataset[args.index]
    mapped = GiTBenchInputs(model_type=model_lib.ModelType.PI05)(
        {
            "image_front": sample["observation.images.front"],
            "image_left": sample["observation.images.left"],
            "image_right": sample["observation.images.right"],
            "state": np.asarray(sample["observation.state"]),
            "actions": np.asarray(sample["action"])[None],
            "prompt": str(sample.get("task", "")),
        }
    )
    assert np.asarray(sample["observation.state"]).shape[-1] == 14
    assert np.asarray(sample["action"]).shape[-1] == 14
    assert all(mapped["image"][key].ndim == 3 for key in mapped["image"])
    assert np.isfinite(np.asarray(sample["observation.state"])).all()
    print(
        {
            "episodes": dataset.num_episodes,
            "samples": len(dataset),
            "state": np.asarray(sample["observation.state"]).shape,
            "actions": np.asarray(sample["action"]).shape,
            "front_rgb": np.asarray(sample["observation.images.front"]).shape,
            "prompt": sample.get("task", ""),
        }
    )


if __name__ == "__main__":
    main()
