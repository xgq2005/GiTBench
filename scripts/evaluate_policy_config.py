#!/usr/bin/env python
"""Launch the standard evaluator from one policy-local YAML configuration."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GITBench policy from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Policy-local YAML configuration, normally policy/<name>/eval_config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and print the resolved evaluator command.",
    )
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    resolved = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay under {ROOT}: {resolved}") from exc
    return resolved


def load_config(path: Path) -> dict[str, Any]:
    path = project_path(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return deepcopy(payload)


def build_eval_command(
    config: Mapping[str, Any],
    *,
    started_at: datetime,
) -> tuple[list[str], Path, dict[str, Any]]:
    policy = _require_string(config, "policy")
    tasks = _string_list(config.get("tasks"), "tasks")
    if not tasks:
        raise ValueError("tasks must contain at least one registered task key")

    split = _string_value(config, "split", "test")
    episodes = _optional_nonnegative_int(config.get("episodes"), "episodes")
    start_episode = _nonnegative_int(config.get("start_episode", 0), "start_episode")
    max_steps = _optional_positive_int(config.get("max_steps"), "max_steps")
    observation = _mapping(config.get("observation", {}), "observation")
    render = _mapping(config.get("render", {}), "render")
    video = _mapping(config.get("video", {}), "video")
    output = _mapping(config.get("output", {}), "output")

    run_dir = _resolve_run_dir(output, started_at)
    log_file = _relative_file_name(output.get("log_file", "log.json"), "output.log_file")
    summary_file = _relative_file_name(
        output.get("summary_file", "summary.json"), "output.summary_file"
    )
    video_dir = _relative_directory_name(
        output.get("video_dir", "videos"), "output.video_dir"
    )

    obs_mode = _string_value(observation, "mode", "rgbd")
    render_mode = _string_value(render, "mode", "rgb_array")
    shader = _string_value(render, "shader", "default")
    sim_backend = _string_value(render, "sim_backend", "auto")
    save_video = _boolean_value(video, "enabled", True)
    video_fps = _positive_int(video.get("fps", 30), "video.fps")
    resume = _boolean_value(config, "resume", True)
    stop_on_success = _boolean_value(config, "stop_on_success", True)
    success_mode = _string_value(config, "success_mode", "ever")
    if success_mode not in {"ever", "final"}:
        raise ValueError("success_mode must be 'ever' or 'final'")
    terminate_on_success = _boolean_value(config, "terminate_on_success", False)

    command = [
        sys.executable,
        str(ROOT / "scripts" / "eval_policy.py"),
        "--policy",
        policy,
        "--tasks",
        *tasks,
        "--split",
        split,
        "--start-episode",
        str(start_episode),
        "--obs-mode",
        obs_mode,
        "--render-mode",
        render_mode,
        "--shader",
        shader,
        "--sim-backend",
        sim_backend,
        "--out",
        str(run_dir / log_file),
        "--summary-out",
        str(run_dir / summary_file),
        "--video-dir",
        str(run_dir / video_dir),
        "--video-fps",
        str(video_fps),
        "--resume" if resume else "--no-resume",
        "--stop-on-success" if stop_on_success else "--no-stop-on-success",
        "--success-mode",
        success_mode,
        "--terminate-on-success" if terminate_on_success else "--no-terminate-on-success",
    ]
    if episodes is not None:
        command.extend(["--episodes", str(episodes)])
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if save_video:
        command.append("--save-video")

    resolved_config = {
        "format": "gitbench_policy_eval_config_v1",
        "source_config": str(project_path(config.get("_config_path", ROOT))),
        "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy": policy,
        "tasks": tasks,
        "split": split,
        "episodes": episodes,
        "start_episode": start_episode,
        "max_steps": max_steps,
        "observation": {"mode": obs_mode},
        "render": {
            "mode": render_mode,
            "shader": shader,
            "sim_backend": sim_backend,
        },
        "video": {
            "enabled": save_video,
            "fps": video_fps,
            "directory": str(video_dir),
        },
        "resume": resume,
        "stop_on_success": stop_on_success,
        "success_mode": success_mode,
        "terminate_on_success": terminate_on_success,
        "output_dir": str(run_dir),
        "command": command,
    }
    return command, run_dir, resolved_config


def _resolve_run_dir(output: Mapping[str, Any], started_at: datetime) -> Path:
    explicit_run_dir = output.get("run_dir")
    if explicit_run_dir is not None:
        if not isinstance(explicit_run_dir, str) or not explicit_run_dir:
            raise ValueError("output.run_dir must be a non-empty project-relative path")
        return project_path(explicit_run_dir)

    root = output.get("root", "eval_results")
    run_name = output.get("run_name")
    if not isinstance(root, str) or not root:
        raise ValueError("output.root must be a non-empty project-relative path")
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("output.run_name must be a non-empty string")
    timestamp = _boolean_value(output, "timestamp", True)
    suffix = started_at.strftime("%Y%m%d_%H%M%S") if timestamp else ""
    directory = f"{run_name}_{suffix}" if suffix else run_name
    return project_path(Path(root) / directory)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_string(config: Mapping[str, Any], name: str) -> str:
    value = config.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_value(config: Mapping[str, Any], name: str, default: str) -> str:
    value = config.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return list(value)


def _boolean_value(config: Mapping[str, Any], name: str, default: bool) -> bool:
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    return None if value is None else _nonnegative_int(value, name)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    return None if value is None else _positive_int(value, name)


def _relative_file_name(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty file name")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay inside the run directory")
    return path


def _relative_directory_name(value: Any, name: str) -> Path:
    path = _relative_file_name(value, name)
    return path


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    config["_config_path"] = str(config_path)
    command, run_dir, resolved_config = build_eval_command(
        config,
        started_at=datetime.now(),
    )
    print(json.dumps(resolved_config, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_config.resolved.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
