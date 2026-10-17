"""GITBench policy adapters for local and OpenPI WebSocket inference."""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class _InferencePolicy(Protocol):
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]: ...


class GITBenchPolicy:
    """Convert benchmark observations to pi05 inputs and cache sampled chunks."""

    def __init__(self, action_space: Any, inference_policy: _InferencePolicy) -> None:
        self.action_space = action_space
        self.inference_policy = inference_policy
        self._chunk: deque[np.ndarray] = deque()
        self._prompt = ""

    def reset(
        self,
        task: Any | None = None,
        episode_id: int | None = None,
        obs: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        del task, episode_id, obs
        self._chunk.clear()
        episode = (info or {}).get("episode", {})
        self._prompt = str(episode.get("instruction", "")) if isinstance(episode, dict) else ""

    def act(self, obs: Any, info: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        if info is not None:
            episode = info.get("episode", {})
            if isinstance(episode, dict) and episode.get("instruction"):
                self._prompt = str(episode["instruction"])
        if not self._chunk:
            result = self.inference_policy.infer(_pi05_observation(obs, self._prompt))
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] < 14:
                raise ValueError(f"Expected pi05 action chunk [H, >=14], got {actions.shape}")
            self._chunk.extend(np.ascontiguousarray(actions[:, :14], dtype=np.float32))
        return _split_action(self._chunk.popleft(), self.action_space)


def make_local_policy(action_space: Any) -> GITBenchPolicy:
    """Create a direct in-process OpenPI policy.

    Required environment variables: ``PI05_CHECKPOINT``. ``PI05_CONFIG``
    defaults to ``pi05_gitbench_lora`` and can be set to ``pi05_gitbench``.
    """
    checkpoint = os.environ.get("PI05_CHECKPOINT")
    if not checkpoint:
        raise ValueError("PI05_CHECKPOINT must point to a trained pi05 checkpoint")
    from openpi.policies import policy_config
    from openpi.training import config

    inference_policy = policy_config.create_trained_policy(
        config.get_config(os.environ.get("PI05_CONFIG", "pi05_gitbench_lora")),
        Path(checkpoint),
    )
    return GITBenchPolicy(action_space, inference_policy)


def make_websocket_policy(action_space: Any) -> GITBenchPolicy:
    """Create a policy backed by OpenPI's original WebSocket client."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    host = os.environ.get("PI05_HOST", "127.0.0.1")
    port = int(os.environ.get("PI05_PORT", "8000"))
    return GITBenchPolicy(action_space, WebsocketClientPolicy(host=host, port=port))


def _pi05_observation(obs: Any, prompt: str) -> dict[str, Any]:
    if not isinstance(obs, dict):
        raise TypeError(f"Expected dict observation, got {type(obs).__name__}")
    sensors = obs.get("sensor_data")
    if not isinstance(sensors, dict):
        raise ValueError("Benchmark observation has no sensor_data mapping")

    front = _camera_rgb(sensors, ("front", "base_camera", "human_cam", "sensor_cam"))
    left = _camera_rgb(sensors, ("left", "left_camera", "left_wrist_camera"), required=False)
    right = _camera_rgb(sensors, ("right", "right_camera", "right_wrist_camera"), required=False)
    state = _dual_arm_qpos(obs)
    return {
        "image_front": front,
        "image_left": front if left is None else left,
        "image_right": front if right is None else right,
        "image_mask": {"front": True, "left": left is not None, "right": right is not None},
        "state": state,
        "prompt": prompt,
    }


def _camera_rgb(sensors: dict[str, Any], names: tuple[str, ...], *, required: bool = True) -> np.ndarray | None:
    ordered = [*names, *(name for name in sensors if name not in names)]
    for name in ordered:
        camera = sensors.get(name)
        if not isinstance(camera, dict):
            continue
        for key in ("rgb", "Color", "color"):
            if key in camera:
                return _uint8_rgb(camera[key])
    if required:
        raise ValueError(f"No RGB image found in sensor cameras {list(sensors)}")
    return None


def _dual_arm_qpos(obs: dict[str, Any]) -> np.ndarray:
    agent = obs.get("agent")
    if not isinstance(agent, dict):
        raise ValueError("Benchmark observation has no multi-agent proprioception")
    try:
        left = _vector(agent["piper_x-0"]["qpos"], 7)
        right = _vector(agent["piper_x-1"]["qpos"], 7)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Expected PiperX qpos for both arms, got {list(agent)}") from exc
    return np.concatenate([left, right], axis=0).astype(np.float32, copy=False)


def _split_action(action: np.ndarray, action_space: Any) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size != 14:
        raise ValueError(f"Expected 14D dual-arm action, got {action.shape}")
    result = {"piper_x-0": action[:7].copy(), "piper_x-1": action[7:].copy()}
    spaces = getattr(action_space, "spaces", {})
    for key, value in result.items():
        space = spaces.get(key) if hasattr(spaces, "get") else None
        if space is not None:
            value[:] = np.clip(value, np.asarray(space.low), np.asarray(space.high))
    return result


def _vector(value: Any, length: int) -> np.ndarray:
    for attribute in ("detach", "cpu", "numpy"):
        method = getattr(value, attribute, None)
        if callable(method):
            value = method()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size < length:
        raise ValueError(f"Expected at least {length} values, got {vector.size}")
    return vector[:length]


def _uint8_rgb(value: Any) -> np.ndarray:
    for attribute in ("detach", "cpu", "numpy"):
        method = getattr(value, attribute, None)
        if callable(method):
            value = method()
    image = np.asarray(value)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image [H, W, C], got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three image channels, got {image.shape}")
    if image.dtype != np.uint8:
        high = float(image.max(initial=0))
        image = np.clip(image, 0.0, 1.0) * 255.0 if high <= 1.0 else np.clip(image, 0, 255)
    return np.ascontiguousarray(image, dtype=np.uint8)
