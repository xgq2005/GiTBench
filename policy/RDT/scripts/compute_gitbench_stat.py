"""Write sparse RDT statistics from the LeRobot v3 GITBench metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from configs.state_vec import STATE_VEC_IDX_MAPPING


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot_root", default="/data/wangyi/xgq/GITBenchv4/data/GiTBench_train")
    parser.add_argument("--save_path", default="configs/dataset_stat.json")
    args = parser.parse_args()
    metadata = json.loads((Path(args.lerobot_root) / "meta/stats.json").read_text())
    entry = metadata["observation.state"]
    indices = [*(STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(7)), *(STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7))]
    mean = np.zeros(128, dtype=np.float64)
    mean[indices] = np.asarray(entry["mean"], dtype=np.float64)
    std = np.zeros(128, dtype=np.float64)
    std[indices] = np.asarray(entry["std"], dtype=np.float64)
    result = {
        "dataset_name": "gitbench",
        "state_mean": mean.tolist(),
        "state_std": std.tolist(),
        "state_min": np.zeros(128, dtype=np.float64).tolist(),
        "state_max": np.zeros(128, dtype=np.float64).tolist(),
    }
    result["state_min"] = np.asarray(result["state_min"], dtype=np.float64)
    result["state_max"] = np.asarray(result["state_max"], dtype=np.float64)
    result["state_min"][indices] = np.asarray(entry["min"], dtype=np.float64)
    result["state_max"][indices] = np.asarray(entry["max"], dtype=np.float64)
    result["state_min"] = result["state_min"].tolist()
    result["state_max"] = result["state_max"].tolist()
    output = Path(args.save_path)
    existing = json.loads(output.read_text()) if output.exists() else {}
    existing[result["dataset_name"]] = result
    output.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"Saved GITBench statistics to {output}")


if __name__ == "__main__":
    main()
