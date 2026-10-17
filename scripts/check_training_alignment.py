"""Audit the converted LeRobot SIM dataset without loading video files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path("/data0/Mem_dataset/yangyang/lerobot_v30/sim/train")
ATOL = 1e-6


def _arr(table, name: str, dtype=None):
    values = table.column(name).to_pylist()
    return np.asarray(values, dtype=dtype)


def main() -> None:
    episodes = {}
    episode_files = sorted((ROOT / "meta/episodes").glob("chunk-*/file-*.parquet"))
    for path in episode_files:
        for row in pq.read_table(path).to_pylist():
            episodes[int(row["episode_index"])] = row
    demo_ends = {}
    with (ROOT / "meta/benchmark_episodes.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            demo_ends[int(row["episode_index"])] = int(row["demo_end"])

    columns = [
        "observation.state", "action", "observation.joint_position.left",
        "observation.joint_position.right", "action.joint_position",
        "action.joint_position.left",
        "action.joint_position.right", "episode_index", "index", "frame_index",
        "timestamp", "benchmark.source_frame", "benchmark.demo_phase",
    ]
    counts = {
        "rows": 0, "episodes": 0, "state_concat_bad": 0, "action_concat_bad": 0,
        "action_alias_bad": 0, "next_state_bad": 0, "last_action_bad": 0,
        "frame_index_bad": 0, "index_bad": 0, "timestamp_bad": 0,
        "source_frame_bad": 0, "demo_phase_bad": 0, "shape_bad": 0,
    }
    max_err = {key: 0.0 for key in (
        "state_concat", "action_concat", "action_alias", "next_state", "last_action",
        "timestamp",
    )}
    first_bad = {}
    previous = None
    global_row = 0
    current_episode = None
    current_frame = -1
    for path in sorted((ROOT / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=columns)
        values = {name: _arr(table, name) for name in columns}
        for i in range(table.num_rows):
            episode = int(values["episode_index"][i])
            frame = int(values["frame_index"][i])
            state = np.asarray(values["observation.state"][i], dtype=np.float64)
            action = np.asarray(values["action"][i], dtype=np.float64)
            state_left = np.asarray(values["observation.joint_position.left"][i], dtype=np.float64)
            state_right = np.asarray(values["observation.joint_position.right"][i], dtype=np.float64)
            action_left = np.asarray(values["action.joint_position.left"][i], dtype=np.float64)
            action_right = np.asarray(values["action.joint_position.right"][i], dtype=np.float64)
            if state.shape != (14,) or action.shape != (14,):
                counts["shape_bad"] += 1
            state_expected = np.concatenate([state_left, state_right])
            action_expected = np.concatenate([action_left, action_right])
            checks = (
                ("state_concat", state, state_expected, "state_concat_bad"),
                ("action_concat", action, action_expected, "action_concat_bad"),
                ("action_alias", action, np.asarray(values["action.joint_position"][i], dtype=np.float64), "action_alias_bad"),
            )
            for label, actual, expected, counter in checks:
                err = float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else float("inf")
                max_err[label] = max(max_err[label], err)
                if err > ATOL:
                    counts[counter] += 1
                    first_bad.setdefault(counter, (str(path), global_row, episode, frame, err))
            if current_episode != episode:
                if current_episode is not None and previous is not None:
                    prev_ep, prev_frame, prev_state, prev_action, _ = previous
                    err = float(np.max(np.abs(prev_action - prev_state)))
                    max_err["last_action"] = max(max_err["last_action"], err)
                    if err > ATOL:
                        counts["last_action_bad"] += 1
                        first_bad.setdefault("last_action_bad", (str(path), global_row - 1, prev_ep, prev_frame, err))
                counts["episodes"] += 1
                current_episode = episode
                current_frame = -1
                previous = None
            if frame != current_frame + 1:
                counts["frame_index_bad"] += 1
                first_bad.setdefault("frame_index_bad", (str(path), global_row, episode, frame, current_frame))
            current_frame = frame
            if int(values["index"][i]) != global_row:
                counts["index_bad"] += 1
                first_bad.setdefault("index_bad", (str(path), global_row, episode, frame, int(values["index"][i])))
            source_err = abs(int(values["benchmark.source_frame"][i]) - frame)
            if source_err:
                counts["source_frame_bad"] += 1
                first_bad.setdefault("source_frame_bad", (str(path), global_row, episode, frame, int(values["benchmark.source_frame"][i])))
            expected_demo = frame <= demo_ends[episode]
            if bool(values["benchmark.demo_phase"][i]) != expected_demo:
                counts["demo_phase_bad"] += 1
                first_bad.setdefault("demo_phase_bad", (str(path), global_row, episode, frame, demo_ends[episode]))
            if previous is not None:
                prev_ep, prev_frame, prev_state, prev_action, _ = previous
                if prev_ep == episode and prev_frame + 1 == frame:
                    err = float(np.max(np.abs(prev_action - state)))
                    max_err["next_state"] = max(max_err["next_state"], err)
                    if err > ATOL:
                        counts["next_state_bad"] += 1
                        first_bad.setdefault("next_state_bad", (str(path), global_row, episode, frame, err))
            dt = None
            if previous is not None and previous[0] == episode:
                dt = float(values["timestamp"][i] - previous[4])
            if dt is not None:
                err = abs(dt - 1.0 / 30.0)
                max_err["timestamp"] = max(max_err["timestamp"], err)
                if err > 2e-5:
                    counts["timestamp_bad"] += 1
                    first_bad.setdefault("timestamp_bad", (str(path), global_row, episode, frame, dt))
            previous = (episode, frame, state, action, float(values["timestamp"][i]))
            global_row += 1
    if previous is not None:
        _, frame, state, action, _ = previous
        err = float(np.max(np.abs(action - state)))
        max_err["last_action"] = max(max_err["last_action"], err)
        if err > ATOL:
            counts["last_action_bad"] += 1
            first_bad.setdefault("last_action_bad", ("<eof>", global_row - 1, current_episode, frame, err))
    counts["rows"] = global_row
    print(json.dumps({"counts": counts, "max_error": max_err, "first_bad": first_bad}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
