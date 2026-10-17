"""GITBench policy adapter for local RDT inference."""

from __future__ import annotations

from collections import deque
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# The Mem-sim checkpoint was trained with a 30 Hz control token and with the
# RDT's sparse state convention is right arm then left arm. GITBench exposes
# the corresponding robots as piper_x-1 (right) and piper_x-0 (left).
RDT_CONTROL_FREQUENCY = 30
# PiperX exposes one finger position in qpos, while the RDT training state
# stores the combined opening of both fingers. The range is 0..0.08 m.
SIM_SINGLE_GRIPPER_WIDTH_MAX = 0.05
RDT_TOTAL_GRIPPER_WIDTH_MAX = 0.08


class GITBenchPolicy:
    def __init__(self, action_space: Any) -> None:
        root = Path(__file__).resolve().parents[1]
        default_checkpoint = _latest_checkpoint(root / "checkpoints" / "rdt-mem-sim")
        checkpoint = Path(os.environ.get("RDT_CHECKPOINT", default_checkpoint))
        if not checkpoint.exists():
            raise FileNotFoundError(f"RDT_CHECKPOINT does not exist: {checkpoint}")
        # Import torch only once an actual RDT policy is requested.  This
        # keeps ``rdt:make_policy`` discoverable in lightweight environments,
        # while avoiding the previous NameError when selecting the dtype.
        import torch

        config_path = Path(os.environ.get("RDT_CONFIG", root / "configs" / "gitbench.yaml"))
        with config_path.open("r") as handle:
            config = yaml.safe_load(handle)
        dtype_name = os.environ.get("RDT_DTYPE", "bf16")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
        # ``eval_policy.py`` itself lives in a top-level ``scripts`` package.
        # RDT's original modules use the same top-level name, so make the local
        # RDT source directory win before importing its model implementation.
        # This keeps the upstream RDT files unchanged while avoiding a package
        # name collision during in-process benchmark evaluation.
        import sys

        rdt_root = str(root)
        if rdt_root in sys.path:
            sys.path.remove(rdt_root)
        sys.path.insert(0, rdt_root)
        existing_scripts = sys.modules.get("scripts")
        existing_path = getattr(existing_scripts, "__path__", None)
        if existing_path is not None and not any(
            str(path).startswith(rdt_root) for path in existing_path
        ):
            sys.modules.pop("scripts", None)

        # Keep RDT's heavy optional imports out of policy discovery. The benchmark
        # only imports the factory before it creates a policy instance.
        from scripts.gitbench_model import GiTBenchRDTModel

        self.model = GiTBenchRDTModel(
            config,
            str(checkpoint),
            os.environ.get("RDT_VISION_ENCODER", str(root / "google" / "siglip-so400m-patch14-384")),
            device=os.environ.get("RDT_DEVICE", "cuda"),
            dtype=dtype,
        )
        self._torch = torch
        self.action_space = action_space
        self.embed_dir = Path(os.environ.get("RDT_LANG_EMBED_DIR", root / "outs" / "gitbench_lang_embeds"))
        self.images: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=2)
        self.actions: deque[np.ndarray] = deque()
        self.instruction = ""
        self.embedding = None

    def reset(self, task=None, episode_id=None, obs=None, info=None) -> None:
        del task, episode_id, obs
        self.images.clear()
        self.actions.clear()
        episode = (info or {}).get("episode", {})
        self.instruction = str(episode.get("instruction", "")) if isinstance(episode, dict) else ""
        self.embedding = self._load_embedding(self.instruction) if self.instruction else None

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None):
        if info:
            episode = info.get("episode", {})
            candidate = str(episode.get("instruction", "")) if isinstance(episode, dict) else ""
            if candidate and candidate != self.instruction:
                self.instruction, self.embedding = candidate, self._load_embedding(candidate)
        current = _observation(obs)
        self.images.append(current[:3])
        if self.embedding is None:
            self.embedding = self._load_embedding(self.instruction)
        if not self.actions:
            previous = self.images[0]
            latest = self.images[-1]
            images = [previous[0], previous[2], previous[1], latest[0], latest[2], latest[1]]
            self.actions.extend(
                self.model.step(
                    current[3],
                    images,
                    self.embedding,
                    control_frequency=RDT_CONTROL_FREQUENCY,
                )
            )
        action = self.actions.popleft()
        return _split_action(action, self.action_space)

    def _load_embedding(self, instruction: str):
        digest = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:12]
        path = self.embed_dir / f"{digest}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding {path}. Run evaluate.sh before rollout.")
        return self._torch.load(path, map_location="cpu")


def _latest_checkpoint(checkpoint_root: Path) -> Path:
    numbered = []
    for path in checkpoint_root.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        weights = path / "pytorch_model.bin"
        if weights.is_file() and weights.stat().st_size > 0:
            numbered.append((step, path))
    if numbered:
        return max(numbered)[1]
    if (checkpoint_root / "pytorch_model.bin").is_file():
        return checkpoint_root
    return checkpoint_root


def _observation(obs: dict[str, Any]):
    sensors = obs.get("sensor_data", {})
    if not isinstance(sensors, dict):
        raise ValueError("Benchmark observation has no sensor_data")
    # These are the keys emitted by ManiSkill's dual-Piper sensor config.  Do
    # not use an arbitrary-sensor fallback here: before these explicit names
    # were included both wrist views silently became the front camera.
    front = _camera(sensors, ("base_camera", "front", "human_cam", "sensor_cam"))
    left = _camera(
        sensors,
        ("piper_x-0-hand_camera", "left", "left_camera", "left_wrist_camera"),
    )
    right = _camera(
        sensors,
        ("piper_x-1-hand_camera", "right", "right_camera", "right_wrist_camera"),
    )
    agent = obs.get("agent", {})
    try:
        left_state = _array(agent["piper_x-0"]["qpos"], 7)
        right_state = _array(agent["piper_x-1"]["qpos"], 7)
        # The environment exposes one finger displacement, but training
        # states store the total opening of both fingers in metres.  Convert
        # qpos -> training units explicitly (rather than normalizing to [0,1])
        # so the model sees the same values it saw during training.
        right_state[6] = np.clip(
            right_state[6] * 2.0, 0.0, RDT_TOTAL_GRIPPER_WIDTH_MAX
        )
        left_state[6] = np.clip(
            left_state[6] * 2.0, 0.0, RDT_TOTAL_GRIPPER_WIDTH_MAX
        )
        state = np.concatenate([left_state, right_state])
    except (KeyError, TypeError) as exc:
        raise ValueError("Expected qpos for piper_x-0 and piper_x-1") from exc
    return front, left, right, state.astype(np.float32)


def _camera(sensors, names, fallback=None):
    for name in names:
        value = sensors.get(name)
        if not isinstance(value, dict):
            continue
        for key in ("rgb", "Color", "color"):
            if key in value:
                image = _array(value[key])
                if image.ndim == 4:
                    image = image[0]
                if image.shape[-1] == 4:
                    image = image[..., :3]
                return np.ascontiguousarray(np.clip(image, 0, 255), dtype=np.uint8)
    if fallback is not None:
        return fallback
    raise ValueError(f"No RGB image in sensor_data keys {list(sensors)}")


def _array(value, length=None):
    for name in ("detach", "cpu", "numpy"):
        method = getattr(value, name, None)
        if callable(method):
            value = method()
    array = np.asarray(value, dtype=np.float32)
    if length is not None:
        array = array.reshape(-1)
        if array.size < length:
            raise ValueError(f"Expected {length} values, got {array.size}")
        return array[:length]
    return array


def _split_action(action, action_space):
    action = np.asarray(action, dtype=np.float32).reshape(14)
    # GITBench training and RDT output use left-arm then right-arm order.
    left = action[:7].copy()
    right = action[7:].copy()
    result = {"piper_x-0": left, "piper_x-1": right}
    # RDT predicts total opening in metres (0..0.08). ManiSkill's
    # normalized position controller expects [-1, 1].
    for value in result.values():
        value[6] = (
            2.0
            * np.clip(value[6] / RDT_TOTAL_GRIPPER_WIDTH_MAX, 0.0, 1.0)
            - 1.0
        )
    spaces = getattr(action_space, "spaces", {})
    for name, value in result.items():
        space = spaces.get(name) if hasattr(spaces, "get") else None
        if space is not None:
            value[:] = np.clip(value, np.asarray(space.low), np.asarray(space.high))
    return result
