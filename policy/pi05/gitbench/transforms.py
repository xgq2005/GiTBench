"""GITBench-to-pi0.5 data mappings shared by training and inference."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class GiTBenchInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front = _rgb(data["image_front"])
        left = _rgb(data["image_left"])
        right = _rgb(data["image_right"])
        masks = data.get("image_mask", {})
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.bool_(masks.get("front", True)),
                "left_wrist_0_rgb": np.bool_(masks.get("left", True)),
                "right_wrist_0_rgb": np.bool_(masks.get("right", True)),
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = str(data["prompt"])
        return inputs


@dataclasses.dataclass(frozen=True)
class GiTBenchOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"], dtype=np.float32)[:, :14]}


@dataclasses.dataclass(frozen=True)
class GiTBenchTransformFactory:
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, model_config: _model.BaseModelConfig) -> transforms.Group:
        return transforms.Group(
            inputs=[GiTBenchInputs(model_type=model_config.model_type)],
            outputs=[GiTBenchOutputs()],
        )


def _rgb(image: object) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (3, 4):
        array = einops.rearrange(array, "c h w -> h w c")
    if array.ndim != 3:
        raise ValueError(f"Expected RGB image [H, W, C], got {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected three image channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(array, dtype=np.uint8)
