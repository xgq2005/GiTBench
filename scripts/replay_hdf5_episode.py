#!/usr/bin/env python3
"""Replay the post-demonstration part of a Mem/GITBench HDF5 episode.

The Mem recorder stores one 7-vector per arm.  The first six values are Piper
joint positions and the last value is the *total* opening of the two fingers
in metres.  GITBench's PiperX controller exposes one normalized mimic-joint
action, so the gripper conversion is explicit below.

By default the script renders frame ``demo_end + 1`` first and then advances
one control step for every following HDF5 frame.  This gives one output frame
per source timestamp (30 Hz for the supplied episode), with the demonstration
frames omitted as requested.  ``--inference-noise`` can add small, correlated
action-block errors to bio4's active arm to approximate a model rollout while
keeping the source timing and object alignment intact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gitbench.env import GITBenchEnv
from gitbench.episodes import get_episodes


TOTAL_GRIPPER_WIDTH_MAX = {
    "bio4": 0.10,
}.get("default", 0.08)


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def hdf5_to_env_action(
    row: np.ndarray, *, gripper_width_max: float = 0.08
) -> np.ndarray:
    """Convert one HDF5 arm row to the 7-D PiperX controller action."""
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.size < 7:
        raise ValueError(f"Expected 7 HDF5 values, got {row.shape}")
    result = row[:7].copy()
    # HDF5 stores both fingers' opening; the normalized mimic
    # controller expects -1..1 (closed..open).
    result[6] = 2.0 * np.clip(result[6] / gripper_width_max, 0.0, 1.0) - 1.0
    return result


def hdf5_to_robot_qpos(
    row: np.ndarray, *, gripper_width_max: float = 0.08
) -> np.ndarray:
    """Expand HDF5 [6 joints, total opening] to PiperX's 8 physical joints."""
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.size < 7:
        raise ValueError(f"Expected 7 HDF5 values, got {row.shape}")
    opening = float(np.clip(row[6], 0.0, gripper_width_max)) * 0.5
    return np.asarray(
        [*row[:6], opening, -opening], dtype=np.float32
    )


def _first_stable_close(
    opening: np.ndarray,
    start: int,
    *,
    threshold: float,
    stable_frames: int,
) -> int | None:
    """Return the first interaction frame with a sustained closed gripper."""
    values = np.asarray(opening, dtype=np.float32).reshape(-1)
    if stable_frames < 1:
        raise ValueError("stable_frames must be positive")
    threshold = float(threshold)
    if threshold < 0.0:
        raise ValueError("close threshold must be non-negative")
    start = max(0, int(start))
    last = len(values) - stable_frames + 1
    for index in range(start, max(start, last)):
        if np.all(values[index : index + stable_frames] <= threshold):
            return index
    return None


def _hold_gripper_after_close(
    left: np.ndarray,
    right: np.ndarray,
    start: int,
    *,
    close_threshold: float = 0.015,
    hold_opening: float = 0.010,
    stable_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | None]]:
    """Keep the arm that actually closes on the object closed until the end.

    The source recordings can reopen after contact because their gripper
    controller differs from the current simulator.  We detect a sustained
    close in the post-demo segment and only clamp that arm; the other arm and
    all six arm joints remain exactly aligned with the HDF5 trajectory.
    """
    close_threshold = float(close_threshold)
    hold_opening = float(hold_opening)
    if hold_opening < 0.0 or hold_opening > close_threshold:
        raise ValueError("hold_opening must be in [0, close_threshold]")
    left_out = np.array(left, dtype=np.float32, copy=True)
    right_out = np.array(right, dtype=np.float32, copy=True)
    close_indices: dict[str, int | None] = {}
    for name, values in (("left", left_out), ("right", right_out)):
        close_index = _first_stable_close(
            values[:, 6],
            start,
            threshold=close_threshold,
            stable_frames=stable_frames,
        )
        close_indices[name] = close_index
        if close_index is not None:
            values[close_index:, 6] = hold_opening
    return left_out, right_out, close_indices


def _add_inference_noise(
    left: np.ndarray,
    right: np.ndarray,
    start: int,
    *,
    seed: int = 0,
    scale: float = 1.0,
    chunk_frames: int = 8,
    max_joint_offset: float = 0.008,
    active_threshold: float = 0.02,
    close_threshold: float = 0.015,
    hesitation_frames: float = 3.0,
    settle_frames: int = 42,
    retry_strength: float = 1.0,
    retry_count: int = 2,
    hold_opening: float = 0.010,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Add smooth model-like action errors to active bio4 arm trajectories.

    Model policies generally emit short action chunks. Their errors are thus
    correlated for several control ticks rather than independent white noise.
    In addition to small joint offsets, this adds short phase reversals before
    grasping (brief over-shoot, backtrack, and re-approach), chunk-level action
    latency, and a near-close/re-open retry of the gripper. The perturbation
    fades out at the end so the source's successful final pose remains exact.
    """
    if chunk_frames < 1:
        raise ValueError("noise chunk_frames must be positive")
    if scale < 0.0:
        raise ValueError("noise scale must be non-negative")
    if max_joint_offset < 0.0:
        raise ValueError("max_joint_offset must be non-negative")
    if active_threshold < 0.0:
        raise ValueError("active_threshold must be non-negative")
    if hesitation_frames < 0.0:
        raise ValueError("hesitation_frames must be non-negative")
    if settle_frames < 1:
        raise ValueError("settle_frames must be positive")
    if retry_strength < 0.0:
        raise ValueError("retry_strength must be non-negative")
    if retry_count < 0:
        raise ValueError("retry_count must be non-negative")
    if hold_opening < 0.0 or hold_opening > close_threshold:
        raise ValueError("hold_opening must be in [0, close_threshold]")

    rng = np.random.default_rng(seed)
    left_out = np.array(left, dtype=np.float32, copy=True)
    right_out = np.array(right, dtype=np.float32, copy=True)
    reference_left = np.array(left, dtype=np.float32, copy=True)
    reference_right = np.array(right, dtype=np.float32, copy=True)
    # Per-joint standard deviations in radians.  Wrist joints receive slightly
    # less noise because they have a larger effect on gripper contact geometry.
    base_std = np.asarray([0.0035, 0.0035, 0.0030, 0.0030, 0.0020, 0.0025])
    base_std *= float(scale)
    details: dict[str, object] = {
        "seed": int(seed),
        "scale": float(scale),
        "chunk_frames": int(chunk_frames),
        "max_joint_offset": float(max_joint_offset),
        "active_threshold": float(active_threshold),
        "hesitation_frames": float(hesitation_frames),
        "settle_frames": int(settle_frames),
        "retry_strength": float(retry_strength),
        "retry_count": int(retry_count),
        "arms": {},
    }

    for name, values in (("left", left_out), ("right", right_out)):
        motion_range = np.ptp(values[start:, :6], axis=0)
        active = bool(np.max(motion_range, initial=0.0) >= active_threshold)
        arm_details = {"active": active, "max_offset": 0.0}
        details["arms"][name] = arm_details
        if not active or len(values) <= start:
            continue

        frame_count = len(values) - start
        close_index = _first_stable_close(
            values[:, 6],
            start,
            threshold=close_threshold,
            stable_frames=3,
        )
        close_offset = (
            close_index - start if close_index is not None else int(frame_count * 0.72)
        )
        close_offset = int(np.clip(close_offset, 1, frame_count - 1))

        # Perturb the action *phase* in frames. A negative phase briefly moves
        # back along the recorded approach, then the positive lobe catches up.
        # Tapering around the close event preserves contact geometry.
        frame_positions = np.arange(frame_count, dtype=np.float32)
        phase = np.zeros(frame_count, dtype=np.float32)
        if hesitation_frames > 0.0 and close_offset > 30:
            width = max(5.0, min(18.0, close_offset * 0.10))
            for center, sign in (
                (close_offset * 0.34, -1.0),
                (close_offset * 0.58, 1.0),
            ):
                distance = (frame_positions - center) / width
                phase += (
                    sign
                    * float(hesitation_frames)
                    * np.sin(np.pi * distance)
                    * np.exp(-0.5 * distance * distance)
                ).astype(np.float32)
            taper_start = max(0.0, close_offset - max(12.0, settle_frames * 0.35))
            taper = np.clip((close_offset - frame_positions) / max(1.0, close_offset - taper_start), 0.0, 1.0)
            phase *= taper

        # Deliberately imperfect re-planning near the object: a real policy
        # may briefly overshoot, notice the visual error, and backtrack before
        # trying the same approach again. These lobes are localized well before
        # the true closure and therefore do not move the final grasp target.
        if retry_strength > 0.0 and retry_count:
            approach_fraction = np.linspace(0.25, 0.68, retry_count)
            lobe_width = max(5.0, min(13.0, close_offset * 0.075))
            for attempt, fraction in enumerate(approach_fraction):
                center = float(close_offset) * float(fraction)
                direction = -1.0 if attempt % 2 == 0 else 1.0
                distance = (frame_positions - center) / lobe_width
                lobe = np.sin(np.pi * distance) * np.exp(-0.5 * distance * distance)
                phase += (
                    direction
                    * float(retry_strength)
                    * (4.0 if attempt == 0 else 3.0)
                    * lobe
                ).astype(np.float32)

            # Chunked inference introduces a small, slowly varying prediction
            # delay.  It is zero at the reset and is tapered before closure.
            delay_count = max(2, int(np.ceil(close_offset / max(1, chunk_frames))))
            delay_anchor_x = np.linspace(0.0, float(close_offset), delay_count)
            delay_anchor_y = rng.normal(0.0, 1.1 * float(retry_strength), delay_count)
            delay_anchor_y[0] = 0.0
            delayed_phase = np.interp(frame_positions, delay_anchor_x, delay_anchor_y)
            delayed_phase *= np.clip(
                (close_offset - frame_positions) / max(1.0, close_offset * 0.72),
                0.0,
                1.0,
            )
            phase += delayed_phase.astype(np.float32)

        # A small settling correction immediately after closure mimics a model
        # rechecking contact. It is gone before the final lift/hold segment.
        post_center = min(frame_count - 1, close_offset + 18)
        post_width = max(6.0, min(16.0, settle_frames * 0.35))
        post_distance = (frame_positions - post_center) / post_width
        post_phase = (
            0.35 * float(hesitation_frames)
            + 0.8 * float(retry_strength)
        ) * np.sin(np.pi * post_distance)
        post_phase *= np.exp(-0.5 * post_distance * post_distance)
        post_phase *= (frame_positions >= close_offset).astype(np.float32)
        if settle_frames:
            settle_start = max(close_offset, frame_count - settle_frames)
            post_phase *= np.clip(
                (frame_count - 1 - frame_positions) / max(1.0, frame_count - 1 - settle_start),
                0.0,
                1.0,
            )
        phase += post_phase.astype(np.float32)
        phase[0] = 0.0
        phase[-1] = 0.0

        source_positions = np.clip(frame_positions + phase, 0.0, frame_count - 1.0)
        source_joint_positions = np.arange(frame_count, dtype=np.float32)
        phase_trajectory = np.empty((frame_count, 6), dtype=np.float32)
        for joint in range(6):
            phase_trajectory[:, joint] = np.interp(
                source_positions,
                source_joint_positions,
                values[start:, joint],
            )
        values[start:, :6] = phase_trajectory

        anchor_count = max(2, (frame_count - 1) // chunk_frames + 2)
        anchors = rng.normal(0.0, base_std, size=(anchor_count, 6))
        anchors[0] = 0.0
        if max_joint_offset:
            anchors = np.clip(anchors, -max_joint_offset, max_joint_offset)
        anchor_positions = np.minimum(
            np.arange(anchor_count, dtype=np.float32) * chunk_frames,
            frame_count - 1,
        )
        frame_positions = np.arange(frame_count, dtype=np.float32)
        offsets = np.empty((frame_count, 6), dtype=np.float32)
        for joint in range(6):
            offsets[:, joint] = np.interp(
                frame_positions,
                anchor_positions,
                anchors[:, joint],
            )
        values[start:, :6] += offsets

        # A near-close attempt followed by a reopen is visually distinctive,
        # but stays safely above the true closed threshold. The final grasp
        # remains controlled by the source close event and hold logic.
        if close_index is not None and retry_count and retry_strength > 0.0:
            for attempt in range(retry_count):
                attempt_rel = int(
                    close_offset
                    - (retry_count - attempt) * max(10, int(close_offset * 0.12))
                )
                attempt_rel = int(np.clip(attempt_rel, 8, close_offset - 8))
                attempt_start = start + attempt_rel
                squeeze_frames = max(2, int(round(2 + retry_strength)))
                reopen_frames = max(3, int(round(4 + retry_strength * 2)))
                squeeze_end = min(close_index, attempt_start + squeeze_frames)
                reopen_end = min(close_index, squeeze_end + reopen_frames)
                if squeeze_end > attempt_start:
                    values[attempt_start:squeeze_end, 6] = max(
                        close_threshold * 1.45,
                        0.022,
                    )
                if reopen_end > squeeze_end:
                    values[squeeze_end:reopen_end, 6] = np.maximum(
                        values[squeeze_end:reopen_end, 6],
                        0.045,
                    )

            # Tiny post-contact regrip: loosen for a couple of control ticks,
            # then return to the held 1 cm opening.
            regrip_start = min(len(values), close_index + 4)
            regrip_end = min(len(values), regrip_start + max(2, int(retry_strength * 2)))
            if regrip_end > regrip_start:
                values[regrip_start:regrip_end, 6] = max(
                    hold_opening,
                    min(close_threshold * 0.92, hold_opening + 0.004 * retry_strength),
                )
            if regrip_end < len(values):
                values[regrip_end:, 6] = hold_opening

        # Keep the first action and final settled pose exactly aligned.
        values[start, :6] = values[start, :6] - offsets[0]
        values[-1, :6] = (
            reference_left[-1, :6]
            if name == "left"
            else reference_right[-1, :6]
        )
        arm_details["max_offset"] = float(np.max(np.abs(offsets), initial=0.0))
        arm_details["max_phase_frames"] = float(np.max(np.abs(phase), initial=0.0))
        arm_details["close_index"] = close_index

    return left_out, right_out, details


def _set_robot_qpos(
    bench: GITBenchEnv,
    left: np.ndarray,
    right: np.ndarray,
    *,
    gripper_width_max: float = 0.08,
) -> None:
    """Set both articulations exactly to the source state (including grippers)."""
    robots = getattr(getattr(bench.raw_env, "agent", None), "agents", None)
    if robots is None or len(robots) < 2:
        raise RuntimeError("The current task does not expose two PiperX agents")
    device = getattr(bench.raw_env, "device", torch.device("cpu"))
    for robot, row in zip(robots[:2], (left, right)):
        qpos = torch.as_tensor(
            hdf5_to_robot_qpos(row, gripper_width_max=gripper_width_max),
            dtype=torch.float32,
            device=device,
        )
        robot.reset(qpos)
    # SAPIEN's CUDA articulation wrapper needs an explicit synchronization for
    # rendering immediately after a teleport.
    scene = getattr(bench.raw_env, "scene", None)
    if scene is not None and hasattr(scene, "_gpu_apply_all"):
        try:
            scene._gpu_apply_all()
            scene.px.gpu_update_articulation_kinematics()
            scene._gpu_fetch_all()
        except Exception:
            pass


def _render_rgb(bench: GITBenchEnv) -> np.ndarray:
    image = _array(bench.render())
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise RuntimeError(f"Unexpected render shape: {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.ascontiguousarray(np.clip(image, 0, 255), dtype=np.uint8)


def _resolve_manifest_episode(source: Path, task_key: str) -> int:
    """Resolve ``<task>_episode<record_id>.hdf5`` through the test manifest."""
    match = re.search(r"episode(\d+)", source.stem)
    if match is None:
        raise ValueError(f"Cannot infer source record id from filename: {source.name}")
    record_id = int(match.group(1))
    matches = [
        row for row in get_episodes(task_key, split="test")
        if int(row.get("source", {}).get("record_id", -1)) == record_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {task_key} test manifest row for source.record_id={record_id}, "
            f"found {len(matches)}"
        )
    return int(matches[0]["episode_id"])


def replay(
    source: Path,
    output: Path,
    *,
    fps: int = 30,
    mode: str = "step",
    shader: str = "default",
    hold_gripper: bool = False,
    close_threshold: float = 0.015,
    hold_opening: float = 0.010,
    close_stable_frames: int = 3,
    inference_noise: bool = False,
    noise_seed: int = 0,
    noise_scale: float = 1.0,
    noise_chunk_frames: int = 8,
    noise_max_joint_offset: float = 0.008,
    noise_hesitation_frames: float = 3.0,
    noise_settle_frames: int = 42,
    noise_retry_strength: float = 1.0,
    noise_retry_count: int = 2,
) -> dict[str, object]:
    with h5py.File(source, "r") as h5:
        left = np.asarray(h5["arm/jointStatePosition/puppetLeft"], dtype=np.float32)
        right = np.asarray(h5["arm/jointStatePosition/puppetRight"], dtype=np.float32)
        timestamps = np.asarray(h5["timestamp"], dtype=np.float64)
        demo_end = int(np.asarray(h5["subtask/demo_end"]))

    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 7:
        raise ValueError(f"Unexpected arm shapes: left={left.shape}, right={right.shape}")
    match = re.match(r"([A-Za-z0-9]+)_episode\d+$", source.stem)
    if match is None:
        raise ValueError(f"Cannot infer task key from filename: {source.name}")
    task_key = match.group(1).lower()
    gripper_width_max = 0.10 if task_key == "bio4" else 0.08
    start = demo_end + 1
    if start >= len(left):
        raise ValueError(f"demo_end={demo_end} leaves no interaction frames")
    close_indices: dict[str, int | None] = {"left": None, "right": None}
    if hold_gripper:
        left, right, close_indices = _hold_gripper_after_close(
            left,
            right,
            start,
            close_threshold=close_threshold,
            hold_opening=hold_opening,
            stable_frames=close_stable_frames,
        )
    noise_details: dict[str, object] = {"enabled": False}
    if inference_noise:
        if task_key != "bio4":
            raise ValueError("--inference-noise is currently supported only for bio4")
        left, right, noise_details = _add_inference_noise(
            left,
            right,
            start,
            seed=noise_seed,
            scale=noise_scale,
            chunk_frames=noise_chunk_frames,
            max_joint_offset=noise_max_joint_offset,
            close_threshold=close_threshold,
            hesitation_frames=noise_hesitation_frames,
            settle_frames=noise_settle_frames,
            retry_strength=noise_retry_strength,
            retry_count=noise_retry_count,
            hold_opening=hold_opening,
        )
    episode_id = _resolve_manifest_episode(source, task_key)
    # The source is sampled at 30 Hz.  Warn, but do not silently resample, if
    # a different file is supplied.
    dt = np.diff(timestamps[start:])
    if len(dt) and not np.allclose(np.median(dt), 1.0 / fps, atol=2e-3):
        print(f"[warn] source median dt={np.median(dt):.6f}s, requested fps={fps}")

    output.parent.mkdir(parents=True, exist_ok=True)
    bench = GITBenchEnv(
        task_key,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        shader=shader,
        max_episode_steps=max(1000, len(left) + 10),
        terminate_on_success=False,
    )
    frames: list[np.ndarray] = []
    try:
        # Reset reproduces the current environment's scene/history.  We do not
        # export those demonstration frames; frame `start` is the first output.
        bench.reset(episode_id=episode_id, split="test")
        _set_robot_qpos(
            bench,
            left[start],
            right[start],
            gripper_width_max=gripper_width_max,
        )
        frames.append(_render_rgb(bench))

        for index in range(start + 1, len(left)):
            action = {
                "piper_x-0": hdf5_to_env_action(
                    left[index], gripper_width_max=gripper_width_max
                ),
                "piper_x-1": hdf5_to_env_action(
                    right[index], gripper_width_max=gripper_width_max
                ),
            }
            bench.step(action)
            if mode == "teleport":
                # Keep the source robot trajectory exact even when current
                # physics/controller gains differ from the recording setup.
                _set_robot_qpos(
                    bench,
                    left[index],
                    right[index],
                    gripper_width_max=gripper_width_max,
                )
            frames.append(_render_rgb(bench))
    finally:
        bench.close()

    if len(frames) != len(left) - start:
        raise AssertionError((len(frames), len(left) - start))
    imageio.mimsave(output, frames, fps=fps, codec="libx264", quality=8)
    return {
        "source": str(source),
        "output": str(output),
        "demo_end": demo_end,
        "manifest_episode_id": episode_id,
        "first_source_frame": start,
        "last_source_frame": len(left) - 1,
        "frame_count": len(frames),
        "fps": fps,
        "mode": mode,
        "hold_gripper": hold_gripper,
        "close_threshold": close_threshold if hold_gripper else None,
        "hold_opening": hold_opening if hold_gripper else None,
        "close_stable_frames": close_stable_frames if hold_gripper else None,
        "close_indices": close_indices,
        "inference_noise": noise_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--mode",
        choices=("step", "teleport"),
        default="step",
        help="step follows current physics; teleport additionally enforces exact qpos each frame",
    )
    parser.add_argument("--shader", default="default")
    parser.add_argument(
        "--hold-gripper",
        action="store_true",
        help="After a sustained close in the interaction segment, keep that gripper closed to the end",
    )
    parser.add_argument(
        "--close-threshold",
        type=float,
        default=0.015,
        help="Maximum total opening (metres) considered a closed grasp",
    )
    parser.add_argument(
        "--hold-opening",
        type=float,
        default=0.010,
        help="Total opening (metres) to hold after grasp detection",
    )
    parser.add_argument(
        "--close-stable-frames",
        type=int,
        default=3,
        help="Number of consecutive closed source frames required before holding",
    )
    parser.add_argument(
        "--inference-noise",
        action="store_true",
        help="Add smooth model-like joint action noise (bio4 only)",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Scale of the correlated joint-position perturbation",
    )
    parser.add_argument(
        "--noise-chunk-frames",
        type=int,
        default=8,
        help="Frames between model-like noise updates",
    )
    parser.add_argument(
        "--noise-max-joint-offset",
        type=float,
        default=0.008,
        help="Maximum absolute joint offset in radians",
    )
    parser.add_argument(
        "--noise-hesitation-frames",
        type=float,
        default=3.0,
        help="Amount of short backtrack/re-approach in control frames",
    )
    parser.add_argument(
        "--noise-settle-frames",
        type=int,
        default=42,
        help="Final frames over which model-like corrections fade out",
    )
    parser.add_argument(
        "--noise-retry-strength",
        type=float,
        default=1.0,
        help="Strength of over-shoot/backtrack and near-close retry behavior",
    )
    parser.add_argument(
        "--noise-retry-count",
        type=int,
        default=2,
        help="Number of near-target retry attempts before the final grasp",
    )
    args = parser.parse_args()
    print(
        replay(
            args.source,
            args.output,
            fps=args.fps,
            mode=args.mode,
            shader=args.shader,
            hold_gripper=args.hold_gripper,
            close_threshold=args.close_threshold,
            hold_opening=args.hold_opening,
            close_stable_frames=args.close_stable_frames,
            inference_noise=args.inference_noise,
            noise_seed=args.noise_seed,
            noise_scale=args.noise_scale,
            noise_chunk_frames=args.noise_chunk_frames,
            noise_max_joint_offset=args.noise_max_joint_offset,
            noise_hesitation_frames=args.noise_hesitation_frames,
            noise_settle_frames=args.noise_settle_frames,
            noise_retry_strength=args.noise_retry_strength,
            noise_retry_count=args.noise_retry_count,
        )
    )


if __name__ == "__main__":
    main()
