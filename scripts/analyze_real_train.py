#!/usr/bin/env python3
"""Run policy inference on the real HIVLA training HDF5 files.

This utility never writes to the source dataset.  It builds the same compact
simulator observation expected by the benchmark adapters, then compares the
returned actions with the next joint positions in each demonstration.

Examples (run from ``GITBenchv4``)::

    python scripts/analyze_real_train.py --policy pi05_lerobot --mode independent \
        --max-episodes 1 --out eval_results/real_train_pi05_smoke.json
    python scripts/analyze_real_train.py --policy rdt --mode sequential \
        --max-episodes 5 --out eval_results/real_train_rdt.json

Use the environment belonging to the selected policy.  RDT requires a T5
embedding file for every instruction; generate missing files separately with
``policy/RDT/scripts/encode_hivla_lang.py``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
import json
from pathlib import Path
import re
import random
import sys
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/data0/Mem_dataset/real")
EPISODE_RE = re.compile(r"^(?P<task>.+)_episode(?P<episode>\d+)\.hdf5$")
DIM_NAMES = [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)]


@dataclass(frozen=True)
class Episode:
    path: Path
    task: str
    episode_id: int
    length: int
    instruction: str


class RealHDF5:
    """Small read-only reader preserving the training t -> t+1 convention."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        self.episodes = self._discover()
        self._handles: dict[Path, h5py.File] = {}
        self.state_sum = np.zeros(14, dtype=np.float64)
        self.state_sq_sum = np.zeros(14, dtype=np.float64)
        self.state_count = 0
        self.state_min = np.full(14, np.inf, dtype=np.float32)
        self.state_max = np.full(14, -np.inf, dtype=np.float32)
        self._compute_stats()

    def _discover(self) -> list[Episode]:
        rows: list[Episode] = []
        for path in self.root.glob("*_episode*.hdf5"):
            match = EPISODE_RE.match(path.name)
            if match is None:
                continue
            with h5py.File(path, "r") as handle:
                length = int(handle["size"][()])
                if length < 2:
                    continue
                required = [
                    "instruction",
                    "arm/jointStatePosition/puppetLeft",
                    "arm/jointStatePosition/puppetRight",
                    "camera/color/front",
                    "camera/color/left",
                    "camera/color/right",
                ]
                missing = [key for key in required if key not in handle]
                if missing:
                    raise ValueError(f"{path} is missing datasets: {missing}")
                for key in required[1:]:
                    if len(handle[key]) != length:
                        raise ValueError(f"{path} {key} length differs from size={length}")
                for arm in ("puppetLeft", "puppetRight"):
                    shape = handle[f"arm/jointStatePosition/{arm}"].shape
                    if shape != (length, 7):
                        raise ValueError(f"{path} {arm} qpos must be [{length}, 7], got {shape}")
                instruction = handle["instruction"][()]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")
            rows.append(
                Episode(
                    path=path,
                    task=match["task"],
                    episode_id=int(match["episode"]),
                    length=length,
                    instruction=str(instruction),
                )
            )
        rows.sort(key=lambda item: (item.task, item.episode_id, item.path.name))
        if not rows:
            raise FileNotFoundError(f"No '*_episode*.hdf5' files found in {self.root}")
        return rows

    def _compute_stats(self) -> None:
        for episode in self.episodes:
            with h5py.File(episode.path, "r") as handle:
                values = self._read_state(handle, slice(None)).astype(np.float64)
            self.state_sum += values.sum(axis=0)
            self.state_sq_sum += np.square(values).sum(axis=0)
            self.state_count += values.shape[0]
            self.state_min = np.minimum(self.state_min, values.min(axis=0))
            self.state_max = np.maximum(self.state_max, values.max(axis=0))
        mean = self.state_sum / max(self.state_count, 1)
        variance = np.maximum(self.state_sq_sum / max(self.state_count, 1) - np.square(mean), 0.0)
        self.state_scale = np.sqrt(variance).astype(np.float32)
        self.state_scale[self.state_scale < 1e-6] = 1.0

    def _handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None or not handle.id.valid:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    @staticmethod
    def _read_state(handle: h5py.File, index: int | slice) -> np.ndarray:
        left = np.asarray(handle["arm/jointStatePosition/puppetLeft"][index], dtype=np.float32)
        right = np.asarray(handle["arm/jointStatePosition/puppetRight"][index], dtype=np.float32)
        return np.concatenate([left, right], axis=-1)

    @staticmethod
    def _decode_image(value: Any) -> np.ndarray:
        encoded = np.asarray(value, dtype=np.uint8).tobytes()
        with Image.open(BytesIO(encoded)) as image:
            return np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8)

    def sample(self, episode_index: int, timestep: int, horizon: int) -> dict[str, Any]:
        episode = self.episodes[episode_index]
        if timestep < 0 or timestep >= episode.length - 1:
            raise IndexError(f"Invalid sample timestep {timestep} for {episode.path.name}")
        handle = self._handle(episode.path)
        current = self._read_state(handle, timestep)
        targets = np.stack(
            [self._read_state(handle, min(timestep + offset, episode.length - 1)) for offset in range(1, horizon + 1)],
            axis=0,
        )
        images = {
            name: self._decode_image(handle[f"camera/color/{name}"][timestep])
            for name in ("front", "left", "right")
        }
        previous_images = {
            name: self._decode_image(handle[f"camera/color/{name}"][max(0, timestep - 1)])
            for name in ("front", "left", "right")
        }
        return {
            "episode_index": episode_index,
            "task": episode.task,
            "episode_id": episode.episode_id,
            "timestep": timestep,
            "length": episode.length,
            "instruction": episode.instruction,
            "state": current,
            "targets": targets,
            "valid_horizon": min(horizon, episode.length - timestep - 1),
            "images": images,
            "previous_images": previous_images,
            "path": str(episode.path),
        }

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles = {}

    def __del__(self) -> None:
        self.close()


class _ActionSpace:
    """Adapter-compatible action space with no clipping side effects."""

    spaces: dict[str, Any] = {}


def _load_policy(name: str):
    sys.path.insert(0, str(ROOT / "policy"))
    sys.path.insert(0, str(ROOT))
    if name == "pi05_lerobot":
        from pi05_lerobot.hivla.benchmark_policy import GITBenchLeRobotPi05Policy

        return GITBenchLeRobotPi05Policy(_ActionSpace())
    if name == "rdt":
        from RDT.hivla.benchmark_policy import GITBenchPolicy

        return GITBenchPolicy(_ActionSpace())
    raise ValueError(f"Unknown policy {name!r}; choose pi05_lerobot or rdt")


def _observation(sample: dict[str, Any], policy_name: str) -> dict[str, Any]:
    state = np.asarray(sample["state"], dtype=np.float32).copy()
    left, right = state[:7].copy(), state[7:].copy()
    # The simulator exposes one finger position, while both HIVLA adapters
    # consume the HDF5 convention (combined opening of both fingers).  Build a
    # simulator-like observation by converting total opening back to qpos;
    # the RDT adapter multiplies it by two again on the way into the model.
    left[6] *= 0.5
    right[6] *= 0.5
    return {
        "sensor_data": {
            "base_camera": {"rgb": sample["images"]["front"]},
            "piper_x-0-hand_camera": {"rgb": sample["images"]["left"]},
            "piper_x-1-hand_camera": {"rgb": sample["images"]["right"]},
        },
        "agent": {
            "piper_x-0": {"qpos": left},
            "piper_x-1": {"qpos": right},
        },
    }


def _prime_rdt_history(policy: Any, sample: dict[str, Any]) -> None:
    """Match the HDF5 loader's two-frame history for an independent sample."""
    images = sample["previous_images"]
    policy.images.clear()
    policy.images.append((images["front"], images["left"], images["right"]))


def _flatten_action(action: dict[str, Any], policy_name: str) -> np.ndarray:
    """Return predictions in common HDF5 order: left arm followed by right."""
    left = np.asarray(action["piper_x-0"], dtype=np.float32).reshape(7).copy()
    right = np.asarray(action["piper_x-1"], dtype=np.float32).reshape(7).copy()
    if policy_name in {"pi05_lerobot", "rdt"}:
        # Adapter output is simulator-normalized gripper [-1, 1]. Convert it
        # back to the HDF5 total opening (0..0.08 m) for comparison.
        left[6] = 0.08 * np.clip((left[6] + 1.0) * 0.5, 0.0, 1.0)
        right[6] = 0.08 * np.clip((right[6] + 1.0) * 0.5, 0.0, 1.0)
    return np.concatenate([left, right]).astype(np.float32)


def _queue(policy: Any):
    for name in ("_actions", "actions"):
        value = getattr(policy, name, None)
        if value is not None:
            return value
    return None


def _raw_to_common(action: Any, policy_name: str) -> np.ndarray:
    """Convert an adapter queue item (or action dict) to left+right HDF5 order."""
    if isinstance(action, dict):
        return _flatten_action(action, policy_name)
    values = np.asarray(action, dtype=np.float32).reshape(14).copy()
    # RDT's internal queue follows training order right+left. pi05's queue is
    # already left+right after its postprocessor.
    if policy_name == "rdt":
        values = np.concatenate([values[7:], values[:7]])
    # Both model queues contain absolute HDF5-space qpos.  Only the dict
    # returned by _split_action uses simulator-normalized gripper values.
    return values.astype(np.float32)


def _infer_chunk(policy: Any, obs: dict[str, Any], info: dict[str, Any], policy_name: str) -> np.ndarray:
    """Request one chunk, preserving the adapter's image-history side effects."""
    first = _flatten_action(policy.act(obs, info), policy_name)
    queue = _queue(policy)
    if queue is None:
        return first[None]
    remaining = [_raw_to_common(item, policy_name) for item in queue]
    queue.clear()
    return np.stack([first, *remaining], axis=0)


@dataclass
class Accumulator:
    horizon: int
    sum_abs_h: np.ndarray = field(init=False)
    sum_sq_h: np.ndarray = field(init=False)
    sum_err_h: np.ndarray = field(init=False)
    sum_norm_abs_h: np.ndarray = field(init=False)
    count_h: np.ndarray = field(init=False)
    sum_abs_dim: np.ndarray = field(init=False)
    sum_sq_dim: np.ndarray = field(init=False)
    sum_err_dim: np.ndarray = field(init=False)
    count_dim: np.ndarray = field(init=False)
    nonfinite: int = 0
    below_data_min: int = 0
    above_data_max: int = 0
    below_data_min_dim: np.ndarray = field(init=False)
    above_data_max_dim: np.ndarray = field(init=False)
    gripper_out_of_range: int = 0
    pred_min: np.ndarray = field(init=False)
    pred_max: np.ndarray = field(init=False)
    samples: int = 0
    temporal_pairs: int = 0
    delta_abs_sum: np.ndarray = field(init=False)
    delta_sq_sum: np.ndarray = field(init=False)
    target_delta_abs_sum: np.ndarray = field(init=False)
    target_delta_sq_sum: np.ndarray = field(init=False)
    sign_flip_count: np.ndarray = field(init=False)
    sign_comparison_count: np.ndarray = field(init=False)
    target_sign_flip_count: np.ndarray = field(init=False)
    stationary_pred_norms: list[float] = field(default_factory=list)
    stationary_target_norms: list[float] = field(default_factory=list)
    stationary_pred_over_threshold: dict[str, int] = field(default_factory=lambda: {"0.05": 0, "0.1": 0, "0.2": 0, "0.5": 0})
    boundary_pred_norms: list[float] = field(default_factory=list)
    interior_pred_norms: list[float] = field(default_factory=list)
    boundary_target_norms: list[float] = field(default_factory=list)
    interior_target_norms: list[float] = field(default_factory=list)
    stationary_arm_pred_norms: list[float] = field(default_factory=list)
    stationary_arm_target_norms: list[float] = field(default_factory=list)
    stationary_arm_pred_over_threshold: dict[str, int] = field(
        default_factory=lambda: {"0.05": 0, "0.1": 0, "0.2": 0, "0.5": 0}
    )
    boundary_arm_pred_norms: list[float] = field(default_factory=list)
    interior_arm_pred_norms: list[float] = field(default_factory=list)
    boundary_arm_target_norms: list[float] = field(default_factory=list)
    interior_arm_target_norms: list[float] = field(default_factory=list)
    _last_prediction: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_delta: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_target_delta: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.sum_abs_h = np.zeros(self.horizon, dtype=np.float64)
        self.sum_sq_h = np.zeros(self.horizon, dtype=np.float64)
        self.sum_err_h = np.zeros(self.horizon, dtype=np.float64)
        self.sum_norm_abs_h = np.zeros(self.horizon, dtype=np.float64)
        self.count_h = np.zeros(self.horizon, dtype=np.int64)
        self.sum_abs_dim = np.zeros(14, dtype=np.float64)
        self.sum_sq_dim = np.zeros(14, dtype=np.float64)
        self.sum_err_dim = np.zeros(14, dtype=np.float64)
        self.count_dim = np.zeros(14, dtype=np.int64)
        self.pred_min = np.full(14, np.inf, dtype=np.float64)
        self.pred_max = np.full(14, -np.inf, dtype=np.float64)
        self.below_data_min_dim = np.zeros(14, dtype=np.int64)
        self.above_data_max_dim = np.zeros(14, dtype=np.int64)
        self.delta_abs_sum = np.zeros(14, dtype=np.float64)
        self.delta_sq_sum = np.zeros(14, dtype=np.float64)
        self.target_delta_abs_sum = np.zeros(14, dtype=np.float64)
        self.target_delta_sq_sum = np.zeros(14, dtype=np.float64)
        self.sign_flip_count = np.zeros(14, dtype=np.int64)
        self.sign_comparison_count = np.zeros(14, dtype=np.int64)
        self.target_sign_flip_count = np.zeros(14, dtype=np.int64)

    def begin_sequence(self) -> None:
        """Clear temporal state at an episode boundary."""
        self._last_prediction = None
        self._last_target = None
        self._last_delta = None
        self._last_target_delta = None

    def add(self, prediction: np.ndarray, target: np.ndarray, valid: int, data_min: np.ndarray, data_max: np.ndarray, scale: np.ndarray, *, temporal: bool = False, chunk_boundary: bool = False) -> None:
        self.samples += 1
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        self.pred_min = np.minimum(self.pred_min, np.nanmin(prediction, axis=0))
        self.pred_max = np.maximum(self.pred_max, np.nanmax(prediction, axis=0))
        self.nonfinite += int(np.size(prediction) - np.isfinite(prediction).sum())
        self.below_data_min += int(np.sum(prediction < data_min))
        self.above_data_max += int(np.sum(prediction > data_max))
        self.below_data_min_dim += np.sum(prediction < data_min, axis=0).astype(np.int64)
        self.above_data_max_dim += np.sum(prediction > data_max, axis=0).astype(np.int64)
        grippers = np.concatenate([prediction[..., 6].reshape(-1), prediction[..., 13].reshape(-1)])
        self.gripper_out_of_range += int(np.sum((grippers < 0) | (grippers > 0.08)))
        valid = min(int(valid), len(prediction), len(target), self.horizon)
        if valid <= 0:
            return
        error = prediction[:valid] - target[:valid]
        finite = np.isfinite(error)
        for index in range(valid):
            mask = finite[index]
            absolute = np.abs(error[index, mask])
            squared = np.square(error[index, mask])
            self.sum_abs_h[index] += absolute.sum()
            self.sum_sq_h[index] += squared.sum()
            self.sum_err_h[index] += error[index, mask].sum()
            self.sum_norm_abs_h[index] += (absolute / scale[mask]).sum()
            self.count_h[index] += int(mask.sum())
        for dim in range(14):
            values = error[:, dim][finite[:, dim]]
            self.sum_abs_dim[dim] += np.abs(values).sum()
            self.sum_sq_dim[dim] += np.square(values).sum()
            self.sum_err_dim[dim] += values.sum()
            self.count_dim[dim] += len(values)

        # In sequential replay each call corresponds to one controller tick.
        # Compare the first action of adjacent calls; independent chunk
        # evaluation deliberately disables this because samples are unrelated.
        if temporal and np.isfinite(prediction[0]).all() and np.isfinite(target[0]).all():
            current_prediction = prediction[0]
            current_target = target[0]
            if self._last_prediction is not None and self._last_target is not None:
                delta = current_prediction - self._last_prediction
                target_delta = current_target - self._last_target
                finite = np.isfinite(delta) & np.isfinite(target_delta)
                if finite.any():
                    self.temporal_pairs += 1
                    self.delta_abs_sum[finite] += np.abs(delta[finite])
                    self.delta_sq_sum[finite] += np.square(delta[finite])
                    self.target_delta_abs_sum[finite] += np.abs(target_delta[finite])
                    self.target_delta_sq_sum[finite] += np.square(target_delta[finite])
                    active = (finite & (np.abs(delta / scale) >= 0.02) & (np.abs(self._last_delta / scale) >= 0.02)) if self._last_delta is not None else np.zeros(14, dtype=bool)
                    target_active = (finite & (np.abs(target_delta / scale) >= 0.02) & (np.abs(self._last_target_delta / scale) >= 0.02)) if self._last_target_delta is not None else np.zeros(14, dtype=bool)
                    self.sign_comparison_count[active] += 1
                    if self._last_delta is not None:
                        self.sign_flip_count[active & (np.sign(delta) != np.sign(self._last_delta))] += 1
                    if self._last_target_delta is not None:
                        self.target_sign_flip_count[target_active & (np.sign(target_delta) != np.sign(self._last_target_delta))] += 1
                    target_norm = float(np.sqrt(np.mean(np.square(target_delta[finite] / scale[finite]))))
                    prediction_norm = float(np.sqrt(np.mean(np.square(delta[finite] / scale[finite]))))
                    (self.boundary_pred_norms if chunk_boundary else self.interior_pred_norms).append(prediction_norm)
                    (self.boundary_target_norms if chunk_boundary else self.interior_target_norms).append(target_norm)
                    arm_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
                    arm_finite = finite[arm_indices]
                    arm_prediction_norm = float(
                        np.sqrt(np.mean(np.square(delta[arm_indices][arm_finite] / scale[arm_indices][arm_finite])))
                    ) if arm_finite.any() else float("nan")
                    arm_target_norm = float(
                        np.sqrt(np.mean(np.square(target_delta[arm_indices][arm_finite] / scale[arm_indices][arm_finite])))
                    ) if arm_finite.any() else float("nan")
                    (self.boundary_arm_pred_norms if chunk_boundary else self.interior_arm_pred_norms).append(arm_prediction_norm)
                    (self.boundary_arm_target_norms if chunk_boundary else self.interior_arm_target_norms).append(arm_target_norm)
                    if arm_target_norm <= 0.05:
                        self.stationary_arm_target_norms.append(arm_target_norm)
                        self.stationary_arm_pred_norms.append(arm_prediction_norm)
                        for threshold in (0.05, 0.1, 0.2, 0.5):
                            if arm_prediction_norm > threshold:
                                self.stationary_arm_pred_over_threshold[str(threshold)] += 1
                    if target_norm <= 0.05:
                        self.stationary_target_norms.append(target_norm)
                        self.stationary_pred_norms.append(prediction_norm)
                        for threshold in (0.05, 0.1, 0.2, 0.5):
                            if prediction_norm > threshold:
                                self.stationary_pred_over_threshold[str(threshold)] += 1
                self._last_delta = delta
                self._last_target_delta = target_delta
            self._last_prediction = current_prediction.copy()
            self._last_target = current_target.copy()

    @staticmethod
    def _metric(abs_sum: np.ndarray, sq_sum: np.ndarray, err_sum: np.ndarray, count: np.ndarray) -> dict[str, list[float]]:
        denominator = np.maximum(count, 1)
        return {
            "count": count.astype(int).tolist(),
            "mae": (abs_sum / denominator).tolist(),
            "rmse": np.sqrt(sq_sum / denominator).tolist(),
            "bias": (err_sum / denominator).tolist(),
        }

    def finish(self, scale: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> dict[str, Any]:
        horizon = self._metric(self.sum_abs_h, self.sum_sq_h, self.sum_err_h, self.count_h)
        per_dim = self._metric(self.sum_abs_dim, self.sum_sq_dim, self.sum_err_dim, self.count_dim)
        per_dim["normalized_mae"] = (self.sum_abs_dim / np.maximum(self.count_dim, 1) / scale).tolist()
        groups = {}
        for name, indices in {
            "arm_only": np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]),
            "gripper_only": np.array([6, 13]),
        }.items():
            groups[name] = {
                "count": int(self.count_dim[indices].sum()),
                "mae": float(self.sum_abs_dim[indices].sum() / max(self.count_dim[indices].sum(), 1)),
                "rmse": float(np.sqrt(self.sum_sq_dim[indices].sum() / max(self.count_dim[indices].sum(), 1))),
                "bias": float(self.sum_err_dim[indices].sum() / max(self.count_dim[indices].sum(), 1)),
                "normalized_mae": float(
                    (self.sum_abs_dim[indices] / np.maximum(self.count_dim[indices], 1) / scale[indices]).mean()
                ),
            }
        horizon["normalized_mae"] = (
            self.sum_norm_abs_h / np.maximum(self.count_h, 1)
        ).tolist()
        if self.temporal_pairs:
            temporal_count = np.maximum(self.temporal_pairs, 1)
            temporal = {
                "pairs": self.temporal_pairs,
                "predicted_delta": {
                    "mae": (self.delta_abs_sum / temporal_count).tolist(),
                    "rmse": np.sqrt(self.delta_sq_sum / temporal_count).tolist(),
                    "normalized_mae": (self.delta_abs_sum / temporal_count / scale).tolist(),
                },
                "target_delta": {
                    "mae": (self.target_delta_abs_sum / temporal_count).tolist(),
                    "rmse": np.sqrt(self.target_delta_sq_sum / temporal_count).tolist(),
                    "normalized_mae": (self.target_delta_abs_sum / temporal_count / scale).tolist(),
                },
                "predicted_to_target_delta_mae_ratio": (self.delta_abs_sum / np.maximum(self.target_delta_abs_sum, 1e-12)).tolist(),
                "sign_flip": {
                    "threshold_normalized_delta": 0.02,
                    "predicted_count": self.sign_flip_count.tolist(),
                    "target_count": self.target_sign_flip_count.tolist(),
                    "comparison_count": self.sign_comparison_count.tolist(),
                    "predicted_rate": (self.sign_flip_count / np.maximum(self.sign_comparison_count, 1)).tolist(),
                    "target_rate": (self.target_sign_flip_count / np.maximum(self.sign_comparison_count, 1)).tolist(),
                },
                "stationary_target": {
                    "threshold_normalized_rms_delta": 0.05,
                    "pairs": len(self.stationary_pred_norms),
                    "predicted_delta_norm_mean": float(np.mean(self.stationary_pred_norms)) if self.stationary_pred_norms else None,
                    "predicted_delta_norm_p50": float(np.quantile(self.stationary_pred_norms, 0.50)) if self.stationary_pred_norms else None,
                    "predicted_delta_norm_p90": float(np.quantile(self.stationary_pred_norms, 0.90)) if self.stationary_pred_norms else None,
                    "predicted_delta_norm_p95": float(np.quantile(self.stationary_pred_norms, 0.95)) if self.stationary_pred_norms else None,
                    "predicted_delta_norm_max": float(np.max(self.stationary_pred_norms)) if self.stationary_pred_norms else None,
                    "target_delta_norm_mean": float(np.mean(self.stationary_target_norms)) if self.stationary_target_norms else None,
                    "predicted_over_threshold_count": self.stationary_pred_over_threshold,
                    "predicted_over_threshold_rate": {
                        key: value / max(len(self.stationary_pred_norms), 1)
                        for key, value in self.stationary_pred_over_threshold.items()
                    },
                },
                "arm_only": {
                    "stationary_target": {
                        "threshold_normalized_rms_delta": 0.05,
                        "pairs": len(self.stationary_arm_pred_norms),
                        "predicted_delta_norm_mean": float(np.mean(self.stationary_arm_pred_norms)) if self.stationary_arm_pred_norms else None,
                        "predicted_delta_norm_p50": float(np.quantile(self.stationary_arm_pred_norms, 0.50)) if self.stationary_arm_pred_norms else None,
                        "predicted_delta_norm_p90": float(np.quantile(self.stationary_arm_pred_norms, 0.90)) if self.stationary_arm_pred_norms else None,
                        "predicted_delta_norm_p95": float(np.quantile(self.stationary_arm_pred_norms, 0.95)) if self.stationary_arm_pred_norms else None,
                        "predicted_delta_norm_max": float(np.max(self.stationary_arm_pred_norms)) if self.stationary_arm_pred_norms else None,
                        "target_delta_norm_mean": float(np.mean(self.stationary_arm_target_norms)) if self.stationary_arm_target_norms else None,
                        "predicted_over_threshold_count": self.stationary_arm_pred_over_threshold,
                        "predicted_over_threshold_rate": {
                            key: value / max(len(self.stationary_arm_pred_norms), 1)
                            for key, value in self.stationary_arm_pred_over_threshold.items()
                        },
                    },
                    "chunk_boundary": {
                        "boundary_pairs": len(self.boundary_arm_pred_norms),
                        "interior_pairs": len(self.interior_arm_pred_norms),
                        "boundary_predicted_delta_norm_mean": float(np.mean(self.boundary_arm_pred_norms)) if self.boundary_arm_pred_norms else None,
                        "interior_predicted_delta_norm_mean": float(np.mean(self.interior_arm_pred_norms)) if self.interior_arm_pred_norms else None,
                        "boundary_target_delta_norm_mean": float(np.mean(self.boundary_arm_target_norms)) if self.boundary_arm_target_norms else None,
                        "interior_target_delta_norm_mean": float(np.mean(self.interior_arm_target_norms)) if self.interior_arm_target_norms else None,
                        "boundary_predicted_delta_norm_p95": float(np.quantile(self.boundary_arm_pred_norms, 0.95)) if self.boundary_arm_pred_norms else None,
                        "interior_predicted_delta_norm_p95": float(np.quantile(self.interior_arm_pred_norms, 0.95)) if self.interior_arm_pred_norms else None,
                    },
                },
                "chunk_boundary": {
                    "boundary_pairs": len(self.boundary_pred_norms),
                    "interior_pairs": len(self.interior_pred_norms),
                    "boundary_predicted_delta_norm_mean": float(np.mean(self.boundary_pred_norms)) if self.boundary_pred_norms else None,
                    "interior_predicted_delta_norm_mean": float(np.mean(self.interior_pred_norms)) if self.interior_pred_norms else None,
                    "boundary_target_delta_norm_mean": float(np.mean(self.boundary_target_norms)) if self.boundary_target_norms else None,
                    "interior_target_delta_norm_mean": float(np.mean(self.interior_target_norms)) if self.interior_target_norms else None,
                    "boundary_predicted_delta_norm_p95": float(np.quantile(self.boundary_pred_norms, 0.95)) if self.boundary_pred_norms else None,
                    "interior_predicted_delta_norm_p95": float(np.quantile(self.interior_pred_norms, 0.95)) if self.interior_pred_norms else None,
                },
            }
        else:
            temporal = {"pairs": 0}
        return {
            "samples": self.samples,
            "horizon": horizon,
            "groups": groups,
            "per_dimension": {name: {key: value[index] for key, value in per_dim.items()} for index, name in enumerate(DIM_NAMES)},
            "checks": {
                "nonfinite_values": self.nonfinite,
                "below_dataset_min_values": self.below_data_min,
                "above_dataset_max_values": self.above_data_max,
                "below_dataset_min_per_dimension": self.below_data_min_dim.tolist(),
                "above_dataset_max_per_dimension": self.above_data_max_dim.tolist(),
                "gripper_out_of_[0,0.08]_values": self.gripper_out_of_range,
                "prediction_min": self.pred_min.tolist(),
                "prediction_max": self.pred_max.tolist(),
                "dataset_min": np.asarray(data_min).tolist(),
                "dataset_max": np.asarray(data_max).tolist(),
            },
            "temporal": temporal,
        }


def _iter_indices(dataset: RealHDF5, max_episodes: int | None, max_samples: int | None, stride: int) -> Iterable[tuple[int, int]]:
    selected = dataset.episodes if max_episodes is None else dataset.episodes[:max_episodes]
    selected_paths = {episode.path for episode in selected}
    count = 0
    for episode_index, episode in enumerate(dataset.episodes):
        if episode.path not in selected_paths:
            continue
        for timestep in range(0, episode.length - 1, stride):
            if max_samples is not None and count >= max_samples:
                return
            yield episode_index, timestep
            count += 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=("pi05_lerobot", "rdt"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=("sequential", "independent"), default="sequential")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=None, help="Comparison horizon; default is 1 sequentially and model chunk length independently.")
    parser.add_argument("--rdt-lang-embed-dir", type=Path, default=None, help="RDT instruction embedding directory; defaults to the real-data cache when present.")
    parser.add_argument("--seed", type=int, default=None, help="Seed numpy/PyTorch diffusion sampling for reproducible outputs.")
    parser.add_argument("--out", type=Path, default=None, help="JSON summary path (default: eval_results/real_train_<policy>_<mode>.json)")
    parser.add_argument("--samples-csv", type=Path, default=None, help="Optional per-sample first-step CSV for downstream plotting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import torch

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except ImportError:
            pass
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.mode == "sequential" and args.stride != 1:
        raise ValueError("sequential mode requires --stride=1 to preserve action-chunk timing")
    out = args.out or ROOT / "eval_results" / f"real_train_{args.policy}_{args.mode}.json"
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_handle = None
    if args.samples_csv:
        csv_path = args.samples_csv if args.samples_csv.is_absolute() else ROOT / args.samples_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_handle = csv_path.open("w", encoding="utf-8")
        csv_handle.write("task,episode_id,timestep,valid_horizon,mae,rmse,bias\n")

    dataset = RealHDF5(args.dataset)
    if args.policy == "rdt":
        embed_dir = args.rdt_lang_embed_dir
        if embed_dir is None:
            candidate = ROOT / "policy" / "RDT" / "outs" / "hivla_lang_embeds_real"
            if candidate.is_dir():
                embed_dir = candidate
        if embed_dir is not None:
            import os

            os.environ["RDT_LANG_EMBED_DIR"] = str(embed_dir.expanduser().resolve())
    policy = _load_policy(args.policy)
    selected = list(_iter_indices(dataset, args.max_episodes, args.max_samples, args.stride))
    if not selected:
        raise RuntimeError("No samples selected")
    requested_horizon = args.horizon or (1 if args.mode == "sequential" else (64 if args.policy == "rdt" else 50))
    overall = Accumulator(requested_horizon)
    per_task: dict[str, Accumulator] = defaultdict(lambda: Accumulator(requested_horizon))
    per_episode: dict[str, Accumulator] = defaultdict(lambda: Accumulator(requested_horizon))
    observed_chunk_lengths: list[int] = []
    processed = 0
    try:
        if args.mode == "independent":
            iterable = selected
            for episode_index, timestep in iterable:
                sample = dataset.sample(episode_index, timestep, requested_horizon)
                obs = _observation(sample, args.policy)
                info = {"episode": {"instruction": sample["instruction"]}}
                policy.reset(episode_id=sample["episode_id"], info=info)
                if args.policy == "rdt":
                    _prime_rdt_history(policy, sample)
                prediction = _infer_chunk(policy, obs, info, args.policy)
                observed_chunk_lengths.append(len(prediction))
                valid = min(sample["valid_horizon"], len(prediction), requested_horizon)
                target = sample["targets"]
                overall.add(prediction, target, valid, dataset.state_min, dataset.state_max, dataset.state_scale)
                key = sample["task"]
                per_task[key].add(prediction, target, valid, dataset.state_min, dataset.state_max, dataset.state_scale)
                episode_key = f"{sample['task']}_episode{sample['episode_id']}"
                per_episode[episode_key].add(prediction, target, valid, dataset.state_min, dataset.state_max, dataset.state_scale)
                if csv_handle:
                    first = prediction[0] - target[0]
                    csv_handle.write(f"{sample['task']},{sample['episode_id']},{timestep},{valid},{np.mean(np.abs(first)):.8g},{np.sqrt(np.mean(np.square(first))):.8g},{np.mean(first):.8g}\n")
                processed += 1
                if processed % 100 == 0:
                    print(f"processed {processed}/{len(selected)} samples", flush=True)
        else:
            # Sequential replay mirrors eval_policy.py: reset once per episode,
            # call act at every timestep, and let each adapter consume its own
            # cached chunk before requesting another one.
            by_episode: dict[int, list[int]] = defaultdict(list)
            for episode_index, timestep in selected:
                by_episode[episode_index].append(timestep)
            for episode_index, timesteps in by_episode.items():
                episode = dataset.episodes[episode_index]
                info = {"episode": {"instruction": episode.instruction}}
                policy.reset(episode_id=episode.episode_id, info=info)
                overall.begin_sequence()
                per_task[episode.task].begin_sequence()
                per_episode[f"{episode.task}_episode{episode.episode_id}"].begin_sequence()
                for timestep in timesteps:
                    sample = dataset.sample(episode_index, timestep, requested_horizon)
                    obs = _observation(sample, args.policy)
                    queue_before = _queue(policy)
                    chunk_boundary = queue_before is not None and len(queue_before) == 0
                    prediction = _flatten_action(policy.act(obs, info if timestep == timesteps[0] else None), args.policy)[None]
                    valid = min(sample["valid_horizon"], 1, requested_horizon)
                    target = sample["targets"]
                    overall.add(prediction, target, valid, dataset.state_min, dataset.state_max, dataset.state_scale, temporal=True, chunk_boundary=chunk_boundary)
                    key = sample["task"]
                    per_task[key].add(prediction, target, valid, dataset.state_min, dataset.state_max, dataset.state_scale, temporal=True, chunk_boundary=chunk_boundary)
                    episode_key = f"{sample['task']}_episode{sample['episode_id']}"
                    per_episode[episode_key].add(prediction, target, valid, dataset.state_min, dataset.state_max, dataset.state_scale, temporal=True, chunk_boundary=chunk_boundary)
                    if csv_handle:
                        first = prediction[0] - target[0]
                        csv_handle.write(f"{sample['task']},{sample['episode_id']},{timestep},{valid},{np.mean(np.abs(first)):.8g},{np.sqrt(np.mean(np.square(first))):.8g},{np.mean(first):.8g}\n")
                    processed += 1
                    if processed % 100 == 0:
                        print(f"processed {processed}/{len(selected)} samples", flush=True)
    finally:
        if csv_handle:
            csv_handle.close()
        dataset.close()

    result = {
        "policy": args.policy,
        "mode": args.mode,
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "episodes_available": len(dataset.episodes),
        "samples_selected": len(selected),
        "samples_processed": processed,
        "requested_horizon": requested_horizon,
        "observed_chunk_lengths": sorted(set(observed_chunk_lengths)),
        "state_scale": dataset.state_scale.tolist(),
        "dimensions": DIM_NAMES,
        "overall": overall.finish(dataset.state_scale, dataset.state_min, dataset.state_max),
        "per_task": {key: value.finish(dataset.state_scale, dataset.state_min, dataset.state_max) for key, value in sorted(per_task.items())},
        "per_episode": {key: value.finish(dataset.state_scale, dataset.state_min, dataset.state_max) for key, value in sorted(per_episode.items())},
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    print(json.dumps({"out": str(out), "samples": processed, "overall_mae": result["overall"]["horizon"]["mae"][0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
