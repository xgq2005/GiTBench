"""Data mapping policy for RoboMME / MemER dataset (panda robot, 2 cameras).

LeRobot dataset keys (flat, no prefix):
  - image            → base_0_rgb
  - wrist_image      → left_wrist_0_rgb
  - state            → state (8-dim, padded to 32)
  - actions          → actions (8-dim, padded to 32)

The language prompt is injected upstream by ``PromptFromLeRobotSubtask``
(``prompt_from_subtask=True`` in DataConfig), which reads ``subtask_index``
from the parquet and looks up the text in ``meta/subtasks.jsonl``.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robomme_example() -> dict:
    """Creates a random input example for the RoboMME policy."""
    return {
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(8),
        "subtask_index": 0,
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboMMEInputs(transforms.DataTransformFn):
    """Map flat LeRobot dataset keys → pi05 model inputs.

    Prompt is expected to be injected upstream by ``PromptFromLeRobotSubtask``
    (set ``prompt_from_subtask=True`` in ``DataConfig``).
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # pi05 has 3 camera slots; pad missing right wrist with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # pi05: mask padded images so they don't contribute to loss
                "right_wrist_0_rgb": np.False_ if self.model_type == _model.ModelType.PI05 else np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass through the prompt if available (injected upstream by PromptFromLeRobotSubtask).
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboMMEOutputs(transforms.DataTransformFn):
    """Slice model output back to dataset action dimension (8)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}


@dataclasses.dataclass(frozen=True)
class RoboMMETransformFactory:
    """GroupFactory for RoboMME transforms. Pass as ``data_transforms`` in SimpleDataConfig."""

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, model_config: _model.BaseModelConfig) -> transforms.Group:
        return transforms.Group(
            inputs=[RoboMMEInputs(model_type=self.model_type)],
            outputs=[RoboMMEOutputs()],
        )
