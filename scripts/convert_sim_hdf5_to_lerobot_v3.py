#!/usr/bin/env python3
"""Convert Mem SIM HDF5 episodes to a LeRobot v3 dataset.

The source files contain a demonstration prefix followed by an execution
segment.  The conversion keeps both parts and uses the next joint-position
sample as the action target::

    observation.state[t] = qpos[t]
    action[t] = qpos[t + 1]

The final source frame has no next-frame action, so its action is set to its
own state (hold position). The execution boundary is retained as the per-frame
``exec_start`` field and as episode metadata, with
``exec_start = demo_end + 1`` in source-frame units.

Videos are encoded as one continuous MP4 per camera.  Episode metadata stores
the timestamp interval belonging to each episode in those files, as required
by the v3 reader.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import av
import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_ROOT = Path("/data0/Mem_dataset/sim/train")
OUTPUT_ROOT = Path("/data0/Mem_dataset/train_lerobot_v3")
FPS = 30
VIDEO_KEYS = ("front", "left", "right")
VIDEO_FEATURE_KEYS = {k: f"observation.images.{k}" for k in VIDEO_KEYS}


def _decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _source_sort_key(path: Path) -> tuple[str, int, str]:
    m = re.match(r"^(.*)_episode(\d+)\.hdf5$", path.name)
    if m:
        return m.group(1), int(m.group(2)), path.name
    return path.stem, -1, path.name


def _iter_source_files() -> list[Path]:
    files = sorted(SOURCE_ROOT.glob("*.hdf5"), key=_source_sort_key)
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found under {SOURCE_ROOT}")
    return files


def _episode_summary(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        n = int(f["size"][()]) if "size" in f else int(f["timestamp"].shape[0])
        demo_end = int(f["subtask/demo_end"][()])
        instruction = _decode_text(f["instruction"][()])
        starts = np.asarray(f["subtask/frame_start"][:], dtype=np.int64)
        ends = np.asarray(f["subtask/frame_end"][:], dtype=np.int64)
        descriptions = [_decode_text(x) for x in f["subtask/description"][:]]
        original_timestamps = np.asarray(f["timestamp"][:], dtype=np.float64)
    if n < 2 or demo_end >= n - 1:
        raise ValueError(f"{path}: no usable execution frames (size={n}, demo_end={demo_end})")
    return {
        "path": path,
        "source_name": path.name,
        "source_episode": int(re.search(r"_episode(\d+)", path.name).group(1))
        if re.search(r"_episode(\d+)", path.name)
        else -1,
        "num_frames": n,
        "demo_end": demo_end,
        "exec_start": demo_end + 1,
        "exec_end_exclusive": n,
        "length": n,
        "instruction": instruction,
        "starts": starts,
        "ends": ends,
        "descriptions": descriptions,
        "original_timestamp_start": float(original_timestamps[0]),
        "original_timestamp_end": float(original_timestamps[-1]),
    }


class VideoEncoder:
    """Small PyAV encoder wrapper; avoids requiring a system ffmpeg binary."""

    def __init__(self, path: Path, width: int, height: int, fps: int = FPS):
        self.fps = fps
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        # NVENC is available on the conversion host.  Keep a CPU fallback for
        # machines where the script is reused without an NVIDIA encoder.
        # Encoding can emit packets only after its internal reorder buffer is
        # full.  PyAV/NVENC occasionally reports an invalid MP4 timestamp when
        # a very short stream is closed after a long-running process, so use a
        # simple, robust CPU H.264 stream here.
        self.stream = self.container.add_stream("libx264", rate=fps, options={"preset": "ultrafast", "crf": "28"})
        self.codec = "libx264"
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = Fraction(1, fps)
        self.frame_count = 0

    def write(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self.frame_count
        frame.time_base = Fraction(1, self.fps)
        self.frame_count += 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def _feature_info(shape: tuple[int, int, int]) -> dict:
    c, h, w = shape
    return {
        "dtype": "video",
        "shape": [c, h, w],
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": h,
            "video.width": w,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": c,
            "has_audio": False,
        },
    }


def _features() -> dict:
    return {
        "observation.images.front": _feature_info((3, 720, 1280)),
        "observation.images.left": _feature_info((3, 480, 640)),
        "observation.images.right": _feature_info((3, 480, 640)),
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)],
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)],
        },
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
        "exec_start": {"dtype": "int64", "shape": [1], "names": None},
    }


def _fixed_list(values: np.ndarray, width: int) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), width)


def _stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _assign_subtasks(summary: dict, source_indices: np.ndarray, subtask_to_index: dict[str, int]) -> np.ndarray:
    result = np.full(source_indices.shape, subtask_to_index["both arms keep static"], dtype=np.int64)
    for start, end, description in zip(summary["starts"], summary["ends"], summary["descriptions"], strict=True):
        mask = (source_indices >= int(start)) & (source_indices <= int(end))
        result[mask] = subtask_to_index[description]
    return result


def _prepare_output() -> None:
    # Preserve unrelated files (notably GITBench's __pycache__) but replace
    # only the generated LeRobot tree if a previous conversion exists.
    for name in ("data", "videos", "meta", "source_manifest.jsonl"):
        target = OUTPUT_ROOT / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    (OUTPUT_ROOT / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    for key in VIDEO_KEYS:
        (OUTPUT_ROOT / f"videos/{VIDEO_FEATURE_KEYS[key]}/chunk-000").mkdir(parents=True, exist_ok=True)


def convert(max_episodes: int | None = None) -> None:
    files = _iter_source_files()
    summaries = [_episode_summary(path) for path in files]
    if max_episodes is not None:
        summaries = summaries[:max_episodes]
    if not summaries:
        raise ValueError("No episodes selected")

    task_names = list(OrderedDict((s["instruction"], None) for s in summaries))
    task_to_index = {name: i for i, name in enumerate(task_names)}
    # Preserve the original annotated-subtask indices and append the fallback
    # used for demo frames and every other unannotated interval.
    subtask_names = list(OrderedDict((d, None) for s in summaries for d in s["descriptions"]))
    if "both arms keep static" not in subtask_names:
        subtask_names.append("both arms keep static")
    subtask_to_index = {name: i for i, name in enumerate(subtask_names)}

    _prepare_output()
    features = _features()
    _write_json(
        OUTPUT_ROOT / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": "bimanual_7dof",
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": len(task_names),
            "chunks_size": 1000,
            "data_files_size_in_mb": 5000,
            "video_files_size_in_mb": 5000,
            "fps": FPS,
            "splits": {"train": f"0:{len(summaries)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                **features,
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "exec_start": {"dtype": "int64", "shape": [1], "names": None},
            },
        },
    )

    pd.DataFrame({"task_index": range(len(task_names))}, index=pd.Index(task_names, name="task")).to_parquet(
        OUTPUT_ROOT / "meta/tasks.parquet"
    )
    pd.DataFrame(
        {"subtask_index": range(len(subtask_names))}, index=pd.Index(subtask_names, name="subtask")
    ).to_parquet(OUTPUT_ROOT / "meta/subtasks.parquet")

    data_path = OUTPUT_ROOT / "data/chunk-000/file-000.parquet"
    data_schema = pa.schema(
        [
            ("timestamp", pa.float32()),
            ("frame_index", pa.int64()),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
            ("subtask_index", pa.int64()),
            ("exec_start", pa.int64()),
            ("observation.state", pa.list_(pa.float32(), 14)),
            ("action", pa.list_(pa.float32(), 14)),
        ]
    )
    data_writer = pq.ParquetWriter(data_path, data_schema, compression="snappy")

    encoders: dict[str, VideoEncoder] = {}
    episodes: list[dict] = []
    manifest_path = OUTPUT_ROOT / "meta/source_manifest.jsonl"
    manifest = manifest_path.open("w", encoding="utf-8")
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_subtasks: list[np.ndarray] = []
    global_index = 0
    video_timestamp = 0.0

    try:
        for episode_index, summary in enumerate(summaries):
            path = summary["path"]
            with h5py.File(path, "r") as f:
                # Keep all frames. For the final frame, action equals state so
                # it represents an explicit hold-position command.
                left_qpos = np.asarray(f["arm/jointStatePosition/puppetLeft"][: summary["num_frames"]], dtype=np.float32)
                right_qpos = np.asarray(f["arm/jointStatePosition/puppetRight"][: summary["num_frames"]], dtype=np.float32)
                next_left = np.concatenate([left_qpos[1:], left_qpos[-1:]], axis=0)
                next_right = np.concatenate([right_qpos[1:], right_qpos[-1:]], axis=0)
                states = np.concatenate([left_qpos, right_qpos], axis=1)
                actions = np.concatenate([next_left, next_right], axis=1)
                n = states.shape[0]
                source_indices = np.arange(0, summary["num_frames"], dtype=np.int64)
                subtask_indices = _assign_subtasks(summary, source_indices, subtask_to_index)

                for key in VIDEO_KEYS:
                    if key not in encoders:
                        first = bytes(np.asarray(f[f"camera/color/{key}"][0], dtype=np.uint8).tobytes())
                        decoded = cv2.imdecode(np.frombuffer(first, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if decoded is None:
                            raise ValueError(f"Cannot decode first {key} frame in {path}")
                        h, w = decoded.shape[:2]
                        encoders[key] = VideoEncoder(
                            OUTPUT_ROOT / f"videos/{VIDEO_FEATURE_KEYS[key]}/chunk-000/file-000.mp4", w, h
                        )

                for local_i, source_i in enumerate(source_indices):
                    for key in VIDEO_KEYS:
                        encoded = bytes(np.asarray(f[f"camera/color/{key}"][int(source_i)], dtype=np.uint8).tobytes())
                        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise ValueError(f"Cannot decode {key} frame {source_i} in {path}")
                        encoders[key].write(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

                batch = {
                    "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS),
                    "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                    "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
                    "index": pa.array(np.arange(global_index, global_index + n, dtype=np.int64)),
                    "task_index": pa.array(np.full(n, task_to_index[summary["instruction"]], dtype=np.int64)),
                    "subtask_index": pa.array(subtask_indices),
                    "exec_start": pa.array(np.full(n, summary["exec_start"], dtype=np.int64)),
                    "observation.state": _fixed_list(states, 14),
                    "action": _fixed_list(actions, 14),
                }
                data_writer.write_table(pa.Table.from_pydict(batch, schema=data_schema))
                # Keep normalization metadata compatible with the previous
                # execution-only dataset. Demo frames are present in Parquet
                # and video, but are deliberately excluded from stats.json.
                stats_start = int(summary["exec_start"])
                # Match the old execution-only stats exactly: exec_start..N-2
                # with next-frame actions. The new final self-action is stored
                # in the dataset but intentionally excluded from stats.json.
                all_states.append(states[stats_start:-1])
                all_actions.append(actions[stats_start:-1])
                all_subtasks.append(subtask_indices[stats_start:-1])

            episode_duration = n / FPS
            episodes.append(
                {
                    "episode_index": episode_index,
                    "tasks": [summary["instruction"]],
                    "length": n,
                    "exec_start": summary["exec_start"],
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": global_index,
                    "dataset_to_index": global_index + n,
                    **{
                        f"videos/{VIDEO_FEATURE_KEYS[key]}/chunk_index": 0
                        for key in VIDEO_KEYS
                    },
                    **{
                        f"videos/{VIDEO_FEATURE_KEYS[key]}/file_index": 0
                        for key in VIDEO_KEYS
                    },
                    **{
                        f"videos/{VIDEO_FEATURE_KEYS[key]}/from_timestamp": video_timestamp
                        for key in VIDEO_KEYS
                    },
                    **{
                        f"videos/{VIDEO_FEATURE_KEYS[key]}/to_timestamp": video_timestamp + episode_duration
                        for key in VIDEO_KEYS
                    },
                    "source_file": summary["source_name"],
                    "source_episode": summary["source_episode"],
                    "source_demo_end": summary["demo_end"],
                    "source_exec_start": summary["exec_start"],
                    "source_exec_end_exclusive": summary["exec_end_exclusive"],
                }
            )
            manifest.write(
                json.dumps(
                    {
                        "episode_index": episode_index,
                        "source_file": summary["source_name"],
                        "source_path": str(path),
                        "source_episode": summary["source_episode"],
                        "num_frames": summary["num_frames"],
                        "demo_end": summary["demo_end"],
                        "exec_start": summary["exec_start"],
                        "exec_end_exclusive": summary["exec_end_exclusive"],
                        "length": n,
                        "original_timestamp_start": summary["original_timestamp_start"],
                        "original_timestamp_end": summary["original_timestamp_end"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            global_index += n
            video_timestamp += episode_duration
            if (episode_index + 1) % 10 == 0 or episode_index == 0:
                print(f"[{episode_index + 1}/{len(summaries)}] {summary['source_name']} frames={n} total={global_index}", flush=True)
    finally:
        manifest.close()
        data_writer.close()
        for encoder in encoders.values():
            encoder.close()

    ep_table = pa.Table.from_pandas(pd.DataFrame(episodes), preserve_index=False)
    pq.write_table(ep_table, OUTPUT_ROOT / "meta/episodes/chunk-000/file-000.parquet", compression="snappy")

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    subtasks = np.concatenate(all_subtasks, axis=0).astype(np.float32).reshape(-1, 1)
    _write_json(
        OUTPUT_ROOT / "meta/stats.json",
        {"observation.state": _stats(states), "action": _stats(actions), "subtask_index": _stats(subtasks)},
    )
    info = json.loads((OUTPUT_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    info["total_episodes"] = len(episodes)
    info["total_frames"] = global_index
    info["total_tasks"] = len(task_names)
    _write_json(OUTPUT_ROOT / "meta/info.json", info)
    print(f"Done: {len(episodes)} episodes, {global_index} frames -> {OUTPUT_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert only the first N episodes (smoke test).")
    args = parser.parse_args()
    convert(args.max_episodes)


if __name__ == "__main__":
    main()
