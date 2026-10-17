"""Data mapping policy for RoboMemArena dataset (franka panda, 2 cameras, LeRobot v3).

LeRobot dataset keys (after repack):
  - image            → observation.images.agentview  (3×256×256, CHW, float32)
  - wrist_image      → observation.images.wrist      (3×256×256, CHW, float32)
  - state            → observation.state             (7-dim: eef_pos + axisangle + gripper)
  - actions          → action                        (7-dim: delta_pose + gripper)

The language prompt is injected upstream by ``PromptFromLeRobotSubtask``
(``prompt_from_subtask=True`` in DataConfig), which reads ``subtask_index``
from the parquet and looks up the text in ``meta/subtasks.parquet``.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robomemarena_example() -> dict:
    """Creates a random input example for the RoboMemArena policy."""
    return {
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(7),
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
class RoboMemArenaInputs(transforms.DataTransformFn):
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
class RoboMemArenaOutputs(transforms.DataTransformFn):
    """Slice model output back to dataset action dimension (7)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class RoboMemArenaTransformFactory:
    """GroupFactory for RoboMemArena transforms. Pass as ``data_transforms`` in SimpleDataConfig."""

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, model_config: _model.BaseModelConfig) -> transforms.Group:
        return transforms.Group(
            inputs=[RoboMemArenaInputs(model_type=self.model_type)],
            outputs=[RoboMemArenaOutputs()],
        )
