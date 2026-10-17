"""Adapt the HAMLET GR00T policy to the GITBench policy protocol.

GITBench exposes one rendered observation and one 7-D qpos vector per PiperX
arm. The GITBench HAMLET checkpoint uses three RGB views plus state and action
groups named ``left_arm`` and ``right_arm``. This adapter executes a
configurable prefix of each predicted action chunk before requesting another.
"""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np


DEFAULT_MEMORY_STRIDE = 50


class GITBenchHAMLETPolicy:
    """Stateful single-episode wrapper around :class:`gr00t.Gr00tPolicy`."""

    def __init__(self, action_space: Any) -> None:
        root = Path(__file__).resolve().parent
        model_path = os.environ.get("HAMLET_MODEL_PATH")
        if not model_path:
            # Keep the checkout usable with the artifacts shipped beside the
            # workspace, while allowing deployments to provide their own path.
            bundled = root / "runs" / "gitbench" / "hamlet_n1d6" / "checkpoint-60000"
            model_path = str(bundled) if bundled.exists() else None
        if not model_path:
            raise ValueError(
                "HAMLET_MODEL_PATH must point to a GR00T/HAMLET checkpoint "
                "(for example .../runs/gitbench/hamlet_n1d6/checkpoint-60000)."
            )
        checkpoint = Path(model_path).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"HAMLET_MODEL_PATH does not exist: {checkpoint}")

        # Import the heavyweight model only after the evaluator has selected
        # this policy. This keeps policy discovery and unit tests lightweight.
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        device = os.environ.get("HAMLET_DEVICE", "cuda")
        strict = _env_bool("HAMLET_STRICT", True)
        self.model = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=str(checkpoint),
            device=device,
            strict=strict,
        )
        _validate_checkpoint_modalities(self.model.modality_configs)
        self.action_space = action_space
        self.exec_chunk = _env_positive_int("HAMLET_EXEC_CHUNK", 16)
        self.demo_prime = _env_bool("HAMLET_DEMO_PRIME", True)
        self.memory_window = _env_positive_int("HAMLET_MEMORY_WINDOW", 4)
        self.memory_stride = _env_positive_int(
            "HAMLET_MEMORY_STRIDE", DEFAULT_MEMORY_STRIDE
        )
        self._actions: deque[np.ndarray] = deque()
        self._instruction = ""
        self._session_id = ""
        self._first_call = True
        self._last_obs: dict[str, Any] | None = None

    def reset(
        self,
        task: Any | None = None,
        episode_id: int | None = None,
        obs: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        del task, episode_id
        self._actions.clear()
        self._session_id = f"gitbench_{uuid.uuid4().hex[:12]}"
        self._first_call = True
        self._last_obs = None
        episode = (info or {}).get("episode", {})
        self._instruction = (
            str(episode.get("instruction", "")) if isinstance(episode, dict) else ""
        )
        if obs is not None:
            self._last_obs = obs
        self._prime_demonstration(info, obs)

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None):
        if not isinstance(obs, dict):
            raise TypeError(f"Expected a dictionary observation, got {type(obs).__name__}")
        self._last_obs = obs
        if info:
            episode = info.get("episode", {})
            if isinstance(episode, dict) and episode.get("instruction"):
                self._instruction = str(episode["instruction"])
        if not self._actions:
            options = {
                "session_ids": [self._session_id],
                "reset_memory": [self._first_call],
            }
            self._first_call = False
            result, _ = self.model.get_action(
                _build_model_observation(obs, self._instruction), options=options
            )
            self._actions.extend(_action_rows(result, self.exec_chunk))
        if not self._actions:
            raise RuntimeError("HAMLET returned an empty action chunk")
        return _split_action(self._actions.popleft(), self.action_space)

    def _prime_demonstration(self, info: dict[str, Any] | None, obs: Any) -> None:
        """Feed approved scene-history frames into HAMLET's rolling cache.

        GITBench deliberately exposes only the priority camera and no historical
        robot state. We therefore reuse the reset state and replicate that camera
        across the model's three view slots. This preserves the no-action demo
        boundary without inventing evaluator-only scene information.
        """
        if not self.demo_prime or not getattr(self.model, "use_hamlet_inference", False):
            return
        demonstration = (info or {}).get("demonstration", {})
        frames = demonstration.get("frames") if isinstance(demonstration, dict) else None
        if not isinstance(frames, np.ndarray) or frames.ndim != 4 or len(frames) <= 1 or obs is None:
            return
        demo_count = int(demonstration.get("demo_frame_count", len(frames) - 1))
        demo_count = min(max(demo_count, 0), len(frames) - 1)
        if demo_count <= 0:
            return
        retained_stride = demonstration.get("demo_frame_stride", 1)
        try:
            retained_stride = max(1, int(retained_stride))
        except (TypeError, ValueError):
            retained_stride = 1
        raw_demo_count = demonstration.get(
            "raw_demo_frame_count", demo_count * retained_stride
        )
        try:
            raw_demo_count = max(0, int(raw_demo_count))
        except (TypeError, ValueError):
            raw_demo_count = demo_count * retained_stride
        indices = _demo_prime_indices(
            demo_count,
            raw_demo_frame_count=raw_demo_count,
            memory_window=self.memory_window,
            memory_stride=self.memory_stride,
            retained_frame_stride=retained_stride,
        )
        for index, frame_index in enumerate(indices):
            prime_obs = _build_model_observation(obs, self._instruction, frame=frames[frame_index])
            self.model.get_action(
                prime_obs,
                options={
                    "session_ids": [self._session_id],
                    "reset_memory": [index == 0],
                    "prime_only": True,
                },
            )
        # The first real observation should append to, rather than replace, the
        # primed cache. The normal act() call consequently must not reset it.
        self._first_call = False


def make_policy(action_space: Any) -> GITBenchHAMLETPolicy:
    return GITBenchHAMLETPolicy(action_space)


def _build_model_observation(
    obs: dict[str, Any], instruction: str, *, frame: np.ndarray | None = None
) -> dict[str, Any]:
    front, left, right = _camera_triplet(obs)
    if frame is not None:
        front = left = right = _uint8_rgb(frame)
    left_arm, right_arm = _dual_arm_state(obs)
    return {
        "video": {
            "front_view": front[None, None, ...],
            "left_wrist_view": left[None, None, ...],
            "right_wrist_view": right[None, None, ...],
        },
        "state": {
            "left_arm": left_arm[None, None, :],
            "right_arm": right_arm[None, None, :],
        },
        "language": {
            "annotation.human.action.task_description": [[instruction]],
        },
    }


def _demo_prime_indices(
    demo_frame_count: int,
    *,
    raw_demo_frame_count: int | None = None,
    memory_window: int,
    memory_stride: int,
    retained_frame_stride: int,
) -> list[int]:
    """Select K-1 retained frames for HAMLET's history window.

    If the raw demonstration is shorter than the requested history span, the
    demo frames and the following interaction frame are spaced uniformly over
    the available history. Otherwise, use ``memory_stride`` as usual.
    """
    if demo_frame_count <= 0 or memory_window <= 1:
        return []
    retained_frame_stride = max(1, retained_frame_stride)
    if raw_demo_frame_count is None:
        raw_demo_frame_count = demo_frame_count * retained_frame_stride
    history_span = (memory_window - 1) * memory_stride
    def retained_index(raw_frame_index: float) -> int:
        return max(
            0,
            min(
                demo_frame_count - 1,
                int(round(raw_frame_index / retained_frame_stride)),
            ),
        )

    if raw_demo_frame_count < history_span:
        return [
            retained_index(
                index * raw_demo_frame_count / (memory_window - 1)
            )
            for index in range(memory_window - 1)
        ]
    return [
        retained_index(
            raw_demo_frame_count
            - (memory_window - 1 - index) * memory_stride
        )
        for index in range(memory_window - 1)
    ]


def _validate_checkpoint_modalities(modality_configs: dict[str, Any]) -> None:
    expected = {
        "video": ["front_view", "left_wrist_view", "right_wrist_view"],
        "state": ["left_arm", "right_arm"],
        "action": ["left_arm", "right_arm"],
        "language": ["annotation.human.action.task_description"],
    }
    actual = {
        name: list(modality_configs[name].modality_keys)
        for name in expected
        if name in modality_configs
    }
    if actual != expected:
        raise ValueError(
            "HAMLET_MODEL_PATH must use the dual-arm GITBench modality layout. "
            f"Expected {expected}, got {actual}."
        )


def _camera_triplet(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensors = obs.get("sensor_data")
    if not isinstance(sensors, dict):
        raise ValueError("Benchmark observation has no sensor_data mapping")
    front = _camera(sensors, ("front", "base_camera", "human_cam", "sensor_cam"))
    left = _camera(
        sensors,
        (
            "left",
            "left_camera",
            "left_wrist_camera",
            "piper_x-0-hand_camera",
            "piper_x-0_hand_camera",
        ),
        fallback=front,
    )
    right = _camera(
        sensors,
        (
            "right",
            "right_camera",
            "right_wrist_camera",
            "piper_x-1-hand_camera",
            "piper_x-1_hand_camera",
        ),
        fallback=front,
    )
    return front, left, right


def _camera(sensors: dict[str, Any], names: tuple[str, ...], fallback: np.ndarray | None = None):
    ordered = [*names, *(name for name in sensors if name not in names)]
    for name in ordered:
        value = sensors.get(name)
        if not isinstance(value, dict):
            continue
        for key in ("rgb", "Color", "color"):
            if key in value:
                return _uint8_rgb(value[key])
    if fallback is not None:
        return fallback
    raise ValueError(f"No RGB image found in sensor_data cameras {list(sensors)}")


def _dual_arm_state(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    agent = obs.get("agent")
    if not isinstance(agent, dict):
        raise ValueError("Benchmark observation has no agent mapping")
    try:
        left = _vector(agent["piper_x-0"]["qpos"], 7)
        right = _vector(agent["piper_x-1"]["qpos"], 7)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Expected qpos for piper_x-0 and piper_x-1, got {list(agent)}") from exc
    return left.astype(np.float32, copy=False), right.astype(np.float32, copy=False)


def _action_rows(actions: dict[str, Any], limit: int) -> list[np.ndarray]:
    try:
        left = np.asarray(actions["left_arm"], dtype=np.float32)
        right = np.asarray(actions["right_arm"], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(
            "HAMLET checkpoint must output left_arm and right_arm action groups"
        ) from exc
    if left.ndim == 3:
        left = left[0]
    if right.ndim == 3:
        right = right[0]
    if left.ndim != 2 or left.shape[1] != 7:
        raise ValueError(f"Expected left_arm action shape [H, 7], got {left.shape}")
    if right.ndim != 2 or right.shape[1] != 7 or len(right) != len(left):
        raise ValueError(f"Expected right_arm action shape [H, 7], got {right.shape}")
    return [
        np.concatenate([left[i], right[i]]).astype(np.float32)
        for i in range(min(limit, len(left)))
    ]


def _split_action(action: np.ndarray, action_space: Any) -> dict[str, np.ndarray]:
    row = np.asarray(action, dtype=np.float32).reshape(-1)
    if row.size != 14:
        raise ValueError(f"Expected 14-D action, got {row.shape}")
    result = {"piper_x-0": row[:7].copy(), "piper_x-1": row[7:].copy()}
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
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected image [H, W, 3/4], got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        high = float(image.max(initial=0))
        image = np.clip(image, 0.0, 1.0) * 255.0 if high <= 1.0 else np.clip(image, 0, 255)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {parsed}")
    return parsed
