#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
POLICY_DIR = ROOT / "policy"
if str(POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(POLICY_DIR))

from gitbench import __version__ as BENCHMARK_VERSION
from gitbench.env import GITBenchEnv
from gitbench.episodes import get_episodes, manifest_path
from gitbench.policy import load_policy
from gitbench.tasks import TASKS, get_task


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a policy on GITBench tasks.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), help="Task keys or env ids.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional limit per task. Default: run every episode in --split.",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Offset within the selected split, useful for debugging.",
    )
    parser.add_argument("--split", default="test", help="Benchmark split to evaluate.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the task catalog step limit. Default: use each task's max_steps.",
    )
    parser.add_argument("--policy", default="zero", help="'zero', 'random', or 'module:factory'.")
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--out", type=Path, default=ROOT / "eval_results" / "log.json")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "eval_results" / "videos")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stop an episode when the benchmark success rule is met. "
            "Use --no-stop-on-success to run until the configured max_steps."
        ),
    )
    parser.add_argument(
        "--success-mode",
        choices=("ever", "final"),
        default="ever",
        help="Benchmark success semantics: latch first success (ever) or use final state.",
    )
    parser.add_argument(
        "--terminate-on-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Make env.step return terminated=True on benchmark success.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read the existing log JSON and skip completed task/episode pairs.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    root = ROOT.resolve()
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Output path must stay under {root}: {resolved}") from exc
    return resolved


def main():
    args = parse_args()
    render_mode = None if str(args.render_mode).lower() in {"none", "null"} else args.render_mode
    if args.save_video and render_mode is None:
        render_mode = "rgb_array"

    args.out = project_path(args.out)
    args.video_dir = project_path(args.video_dir)
    args.summary_out = project_path(args.summary_out) if args.summary_out else args.out.with_name("summary.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_video:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = _run_metadata(args)
    all_results = (
        _load_resumable_results(args.out, run_metadata) if args.resume else []
    )
    completed = _completed_episode_keys(all_results)
    _write_log(args.out, all_results, args, run_metadata)

    for task_name in args.tasks:
        spec = get_task(task_name)
        bench_env = None
        episodes = get_episodes(
            spec.key,
            split=args.split,
            start=args.start_episode,
            limit=args.episodes,
        )
        try:
            max_steps = args.max_steps if args.max_steps is not None else spec.max_steps
            bench_env = GITBenchEnv(
                spec,
                obs_mode=args.obs_mode,
                render_mode=render_mode,
                shader=args.shader,
                sim_backend=args.sim_backend,
                max_episode_steps=max_steps,
                success_mode=args.success_mode,
                terminate_on_success=args.terminate_on_success,
            )
            policy_load_error = None
            try:
                policy = load_policy(args.policy, bench_env.action_space)
            except Exception as exc:
                policy = None
                policy_load_error = repr(exc)
            for episode in episodes:
                episode_id = int(episode["episode_id"])
                if (spec.key, episode_id) in completed:
                    print(json.dumps({"task_key": spec.key, "episode_id": episode_id, "skipped": "already_in_log"}, ensure_ascii=False))
                    continue
                if policy_load_error is not None:
                    video_path = None
                    video_error = None
                    if args.save_video:
                        video_path, video_error = _write_video(
                            [
                                _diagnostic_frame(
                                    spec.key, episode_id, policy_load_error
                                )
                            ],
                            args.video_dir,
                            spec.key,
                            episode_id,
                            args.video_fps,
                            policy_load_error,
                            "fail",
                        )
                    result = {
                        "task_key": spec.key,
                        "env_id": spec.env_id,
                        "gym_env_id": spec.gym_env_id,
                        "source_family": spec.source_family,
                        "episode_id": episode_id,
                        "split": episode["split"],
                        "source_kind": episode["source"]["kind"],
                        "source_record_id": episode["source"]["record_id"],
                        "max_steps": max_steps,
                        "stop_on_success": args.stop_on_success,
                        "success_mode": args.success_mode,
                        "terminate_on_success": args.terminate_on_success,
                        "success": False,
                        "error_kind": "policy",
                        "outcome": "failure",
                        "scorable": spec.scorable,
                        "steps": 0,
                        "terminated": False,
                        "truncated": False,
                        "elapsed_sec": 0.0,
                        "error": policy_load_error,
                        "policy_load_error": policy_load_error,
                        "policy_reset_error": None,
                        "policy_act_error": None,
                        "failure_reason": None,
                        "progress": None,
                        **_demonstration_log_fields({}, None),
                        **_video_log_fields(None),
                        "video_path": str(video_path) if video_path else None,
                        "video_error": video_error,
                    }
                    _record_result(
                        all_results, completed, result, args, run_metadata
                    )
                    continue
                start = time.time()
                video_path = None
                reset_error = None
                policy_reset_error = None
                policy_act_error = None
                recorder = (
                    _EpisodeVideoRecorder(
                        args.video_dir,
                        spec.key,
                        episode_id,
                        args.video_fps,
                    )
                    if args.save_video
                    else None
                )
                if recorder is not None:
                    bench_env.set_demonstration_frame_callback(
                        lambda is_demonstration, env=bench_env, target=recorder: target.capture(
                            env, is_demonstration=is_demonstration
                        )
                    )
                try:
                    obs, info = bench_env.reset(episode_id=episode_id, split=args.split)
                except Exception as exc:
                    obs, info = None, {}
                    reset_error = repr(exc)
                finally:
                    bench_env.set_demonstration_frame_callback(None)

                # ``GITBenchEnv.step`` intentionally exposes only policy-safe
                # fields and therefore drops the reset-time demonstration
                # payload. Keep its compact metadata separately so episode
                # logs still report the v4 raw/retained frame counts after the
                # rollout has started (without serializing the frame array).
                demonstration_info = {
                    "demonstration": info.get("demonstration")
                } if isinstance(info, dict) else {}

                if reset_error is None:
                    try:
                        _reset_policy(policy, spec, episode_id, obs, info)
                    except Exception as exc:
                        policy_reset_error = repr(exc)

                success = bool(info.get("success", False))
                steps = 0
                terminated = truncated = False
                error = reset_error or policy_reset_error
                if error is None:
                    while steps < max_steps and not (terminated or truncated):
                        try:
                            action = policy.act(obs, info)
                            if not bench_env.action_space.contains(action):
                                raise ValueError(
                                    "Policy action is outside the environment action space"
                                )
                        except Exception as exc:
                            policy_act_error = repr(exc)
                            error = policy_act_error
                            break
                        try:
                            obs, reward, terminated, truncated, info = bench_env.step(action)
                            success = bool(info.get("success", False))
                            steps += 1
                        except Exception as exc:
                            error = repr(exc)
                            break
                        if recorder is not None:
                            recorder.capture(bench_env, is_demonstration=False)
                        if success and args.stop_on_success:
                            break

                error_kind = _error_kind(
                    reset_error=reset_error,
                    policy_load_error=None,
                    policy_reset_error=policy_reset_error,
                    policy_act_error=policy_act_error,
                    error=error,
                )
                outcome = _outcome(success, error_kind)
                video_error = None
                if recorder is not None:
                    if recorder.frame_count == 0:
                        diagnostic_error = (
                            recorder.error
                            or error
                            or "No rendered frames were produced."
                        )
                        video_path, video_error = _write_video(
                            [_diagnostic_frame(spec.key, episode_id, diagnostic_error)],
                            args.video_dir,
                            spec.key,
                            episode_id,
                            args.video_fps,
                            diagnostic_error,
                            _video_outcome(outcome),
                        )
                    else:
                        video_path, video_error = recorder.finish(
                            _video_outcome(outcome)
                        )

                result = {
                    "task_key": spec.key,
                    "env_id": spec.env_id,
                    "gym_env_id": spec.gym_env_id,
                    "source_family": spec.source_family,
                    "episode_id": episode_id,
                    "split": episode["split"],
                    "source_kind": episode["source"]["kind"],
                    "source_record_id": episode["source"]["record_id"],
                    "max_steps": max_steps,
                    "stop_on_success": args.stop_on_success,
                    "success_mode": args.success_mode,
                    "terminate_on_success": args.terminate_on_success,
                    # Keep the boolean and categorical result consistent.
                    # In particular, a policy exception invalidates an
                    # already-satisfied reset state and is scored as failure.
                    "success": outcome == "success",
                    "error_kind": error_kind,
                    "outcome": outcome,
                    "scorable": bool(info.get("scorable", spec.scorable)),
                    "steps": steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "elapsed_sec": round(time.time() - start, 4),
                    "error": error,
                    "policy_load_error": None,
                    "policy_reset_error": policy_reset_error,
                    "policy_act_error": policy_act_error,
                    "failure_reason": bench_env.failure_reason,
                    "progress": (
                        bench_env.evaluate().get("progress")
                        if reset_error is None
                        else None
                    ),
                    **_demonstration_log_fields(demonstration_info, reset_error),
                    **_video_log_fields(recorder),
                    "video_path": str(video_path) if video_path else None,
                    "video_error": video_error,
                }
                _record_result(
                    all_results, completed, result, args, run_metadata
                )
        except Exception as exc:
            for episode in episodes:
                episode_id = int(episode["episode_id"])
                if (spec.key, episode_id) in completed:
                    print(json.dumps({"task_key": spec.key, "episode_id": episode_id, "skipped": "already_in_log"}, ensure_ascii=False))
                    continue
                video_path = None
                video_error = None
                if args.save_video:
                    video_path, video_error = _write_video(
                        [_diagnostic_frame(spec.key, episode_id, repr(exc))],
                        args.video_dir,
                        spec.key,
                        episode_id,
                        args.video_fps,
                        repr(exc),
                        "error",
                    )
                result = {
                    "task_key": spec.key,
                    "env_id": spec.env_id,
                    "gym_env_id": spec.gym_env_id,
                    "source_family": spec.source_family,
                    "episode_id": episode_id,
                    "split": episode["split"],
                    "source_kind": episode["source"]["kind"],
                    "source_record_id": episode["source"]["record_id"],
                    "max_steps": args.max_steps if args.max_steps is not None else spec.max_steps,
                    "stop_on_success": args.stop_on_success,
                    "success_mode": args.success_mode,
                    "terminate_on_success": args.terminate_on_success,
                    "success": False,
                    "outcome": "error",
                    "error_kind": "infrastructure",
                    "scorable": spec.scorable,
                    "steps": 0,
                    "terminated": False,
                    "truncated": False,
                    "elapsed_sec": 0.0,
                    "error": repr(exc),
                    "policy_load_error": None,
                    "policy_reset_error": None,
                    "policy_act_error": None,
                    "failure_reason": None,
                    "progress": None,
                    **_demonstration_log_fields({}, repr(exc)),
                    **_video_log_fields(None),
                    "video_path": str(video_path) if video_path else None,
                    "video_error": video_error,
                }
                _record_result(
                    all_results, completed, result, args, run_metadata
                )
        finally:
            if bench_env is not None:
                bench_env.close()

    summary = _summary(all_results, args)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    successes = summary["successes"]
    total = summary["total"]
    rate = summary["success_rate"]
    if rate is None:
        provisional = summary["provisional_success_rate"]
        provisional_text = (
            f"{provisional:.4f}" if provisional is not None else "unavailable"
        )
        print(
            "success_rate=unavailable "
            f"({summary['infrastructure_errors']} infrastructure errors; "
            f"provisional={provisional_text})"
        )
    else:
        print(f"success_rate={rate:.4f} ({successes}/{total})")
    _write_log(args.out, all_results, args, run_metadata)
    print(f"log saved to {args.out}")
    print(f"summary saved to {args.summary_out}")


def _record_result(
    all_results: list[dict[str, Any]],
    completed: set[tuple[str, int]],
    result: dict[str, Any],
    args,
    run_metadata: dict[str, Any],
) -> None:
    all_results.append(result)
    if result.get("error_kind") != "infrastructure":
        completed.add((result["task_key"], int(result["episode_id"])))
    _write_log(args.out, all_results, args, run_metadata)
    print(json.dumps(result, ensure_ascii=False))


def _reset_policy(
    policy: Any,
    spec: Any,
    episode_id: int,
    obs: Any,
    info: dict[str, Any],
) -> None:
    """Call the policy reset hook after the demonstration-bearing env reset.

    Older project-local policies that only accept ``task`` and ``episode_id``
    continue to work. New policies should accept ``obs`` and ``info`` and read
    the scene-only history from ``info["demonstration"]``.
    """
    reset = getattr(policy, "reset", None)
    if not callable(reset):
        return
    supplied = {
        "task": spec,
        "episode_id": episode_id,
        "obs": obs,
        "info": info,
    }
    try:
        signature = inspect.signature(reset)
    except (TypeError, ValueError):
        reset(**supplied)
        return
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = {
        name: value
        for name, value in supplied.items()
        if accepts_kwargs or name in signature.parameters
    }
    reset(**kwargs)


def _demonstration_log_fields(
    info: dict[str, Any], reset_error: str | None
) -> dict[str, Any]:
    demonstration = info.get("demonstration", {})
    if not isinstance(demonstration, dict):
        demonstration = {}
    return {
        "demo_frame_count": demonstration.get("demo_frame_count"),
        "demo_raw_frame_count": demonstration.get("raw_demo_frame_count"),
        "demo_frame_stride": demonstration.get("demo_frame_stride"),
        "demo_total_frame_count": (
            int(len(demonstration["frames"]))
            if demonstration.get("frames") is not None
            else None
        ),
        "demo_interaction_frame_index": demonstration.get("interaction_frame_index"),
        "demo_camera": demonstration.get("camera"),
        "demo_timing_source": demonstration.get("timing_source"),
        "demonstration_error": (
            reset_error
            if reset_error and "DemonstrationReplayError" in reset_error
            else None
        ),
    }


def _completed_episode_keys(results: list[dict[str, Any]]) -> set[tuple[str, int]]:
    completed = set()
    for result in _latest_episode_results(results):
        if _result_outcome(result) == "error":
            continue
        key = _episode_key(result)
        if key is not None:
            completed.add(key)
    return completed


def _load_log(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    if not path.exists():
        return {"results": []}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {"results": []}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_log_results(path: Path) -> list[dict[str, Any]]:
    data = _load_log(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    raise ValueError(f"Unsupported log format in {path}")


def _load_resumable_results(
    path: Path, expected_metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    payload = _load_log(path)
    if not isinstance(payload, dict) or payload.get("format") != "gitbench_eval_log_v2":
        raise ValueError(
            f"Cannot resume legacy or unstructured log {path}; use a new output path"
        )
    actual = payload.get("run_metadata")
    if not isinstance(actual, dict) or actual.get("run_fingerprint") != expected_metadata["run_fingerprint"]:
        raise ValueError(
            f"Cannot resume {path}: policy, scoring options, runtime, or manifest changed"
        )
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Unsupported log results in {path}")
    # Keep failed infrastructure attempts for auditability. Completion and
    # scoring use the latest attempt for each episode, so a retry does not
    # create a duplicate score.
    return [result for result in results if isinstance(result, dict)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(roots: list[Path]) -> str:
    """Hash a small source/config tree with stable project-relative names."""
    # Include executable source and the declarative files that affect task
    # construction/evaluation.  Omitting YAML/TOML made a resume silently
    # accept changed simulator or policy configuration even though the run
    # fingerprint was expected to protect against that.
    included_suffixes = {".py", ".json", ".yaml", ".yml", ".toml"}
    files = sorted(
        {
            path
            for root in roots
            for path in (
                root.rglob("*") if root.is_dir() else [root]
            )
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in included_suffixes
        },
        key=lambda path: str(path.relative_to(ROOT)),
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_metadata(args) -> dict[str, Any]:
    metadata = {
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_source_sha256": _source_tree_sha256(
            [
                ROOT / "gitbench",
                ROOT / "scripts",
                ROOT / "task_config",
                ROOT / "third_party" / "ManiSkill-3" / "mani_skill" / "envs" / "tasks",
            ]
        ),
        "manifest_sha256": _sha256(manifest_path()),
        "policy": args.policy,
        "policy_source_sha256": _policy_source_sha256(args.policy),
        "policy_runtime": _policy_runtime_metadata(args.policy),
        "tasks": list(args.tasks),
        "split": args.split,
        "episodes_per_task": args.episodes,
        "start_episode": args.start_episode,
        "max_steps_override": args.max_steps,
        "stop_on_success": args.stop_on_success,
        "success_mode": getattr(args, "success_mode", "ever"),
        "terminate_on_success": getattr(args, "terminate_on_success", False),
        "obs_mode": args.obs_mode,
        "render_mode": args.render_mode,
        "shader": args.shader,
        "sim_backend": args.sim_backend,
        "python_version": platform.python_version(),
        "torch_version": _package_version("torch"),
        "maniskill_version": _package_version("mani_skill"),
    }
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    metadata["run_fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return metadata


def _policy_source_sha256(policy_spec: str) -> str | None:
    """Hash local policy source without traversing multi-gigabyte weights."""
    if policy_spec in {"zero", "random"} or ":" not in policy_spec:
        return None
    module_name = policy_spec.split(":", 1)[0]
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    roots = [Path(path) for path in spec.submodule_search_locations or []]
    # ``rdt`` is a lower-case compatibility shim for the implementation kept
    # in ``policy/RDT``.  Hash both trees so changing the real adapter changes
    # the resume fingerprint as expected.
    if module_name.lower() == "rdt":
        implementation_root = ROOT / "policy" / "RDT"
        if implementation_root.is_dir() and implementation_root not in roots:
            roots.append(implementation_root)
    if roots:
        files = sorted(
            path
            for root in roots
            for path in root.rglob("*")
            if "__pycache__" not in path.parts
            and path.is_file()
            and path.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
        )
        base = roots[0]
    elif spec.origin and spec.origin not in {"built-in", "frozen"}:
        files = [Path(spec.origin)]
        base = files[0].parent
    else:
        return None
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(base)
        except ValueError:
            relative = path
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest() if files else None


_POLICY_RUNTIME_ENV_KEYS = (
    # Local RDT adapter.
    "RDT_CHECKPOINT",
    "RDT_CONFIG",
    "RDT_DTYPE",
    "RDT_DEVICE",
    "RDT_VISION_ENCODER",
    "RDT_LANG_EMBED_DIR",
    # Local/WebSocket pi05 adapter.
    "PI05_CHECKPOINT",
    "PI05_CONFIG",
    "PI05_BACKEND",
    "PI05_HOST",
    "PI05_PORT",
    # Local SIM pi05 adapter.
    "PI05_SIM_CHECKPOINT",
    "PI05_SIM_EXEC_CHUNK",
    # Local LeRobot pi05 adapter.
    "PI05_LEROBOT_CHECKPOINT",
    "PI05_LEROBOT_DEVICE",
    "PI05_LEROBOT_TOKENIZER",
    # Local HAMLET adapter.
    "HAMLET_MODEL_PATH",
    "HAMLET_DEVICE",
    "HAMLET_STRICT",
    "HAMLET_EXEC_CHUNK",
    "HAMLET_DEMO_PRIME",
    "HAMLET_MEMORY_WINDOW",
    "HAMLET_MEMORY_STRIDE",
)
_POLICY_RUNTIME_PATH_KEYS = {
    "RDT_CHECKPOINT",
    "RDT_CONFIG",
    "RDT_VISION_ENCODER",
    "RDT_LANG_EMBED_DIR",
    "PI05_CHECKPOINT",
    "PI05_SIM_CHECKPOINT",
    "PI05_LEROBOT_CHECKPOINT",
    "HAMLET_MODEL_PATH",
}


def _policy_runtime_metadata(policy_spec: str) -> dict[str, Any]:
    """Capture external policy inputs that source hashing cannot cover.

    Model checkpoints are intentionally not read into the benchmark log.  A
    path, existence bit, size and nanosecond mtime are enough to invalidate a
    resume when a checkpoint is replaced while keeping startup cheap even for
    multi-gigabyte directories.  Environment variables are included only for
    the project adapters that define them, so unrelated process state does not
    make otherwise identical runs incomparable.
    """
    if policy_spec in {"zero", "random"}:
        return {"environment": {}, "paths": {}}
    module_name = policy_spec.split(":", 1)[0].lower()
    # The checked-in RDT adapter is addressed as ``rdt:make_policy`` by the
    # YAML configs, while its package directory is intentionally ``RDT``.
    # Keep the runtime fingerprint independent of import-path casing.
    defaults: dict[str, str] = {}
    if module_name == "rdt":
        relevant_keys = {
            "RDT_CHECKPOINT",
            "RDT_CONFIG",
            "RDT_DTYPE",
            "RDT_DEVICE",
            "RDT_VISION_ENCODER",
            "RDT_LANG_EMBED_DIR",
        }
        rdt_root = ROOT / "policy" / "RDT"
        defaults = {
            "RDT_CHECKPOINT": str(_latest_rdt_checkpoint(rdt_root)),
            "RDT_CONFIG": str(rdt_root / "configs" / "hivla.yaml"),
            "RDT_DTYPE": "bf16",
            "RDT_DEVICE": "cuda",
            "RDT_VISION_ENCODER": str(
                rdt_root / "google" / "siglip-so400m-patch14-384"
            ),
            "RDT_LANG_EMBED_DIR": str(rdt_root / "outs" / "hivla_lang_embeds"),
        }
    elif module_name == "pi05":
        relevant_keys = {
            "PI05_CHECKPOINT",
            "PI05_CONFIG",
            "PI05_BACKEND",
            "PI05_HOST",
            "PI05_PORT",
        }
        defaults = {
            "PI05_CONFIG": "pi05_hivla_lora",
            "PI05_BACKEND": "websocket",
            "PI05_HOST": "127.0.0.1",
            "PI05_PORT": "8000",
        }
    elif module_name in {"pi05_lerobot", "pi05-lerobot"}:
        relevant_keys = {
            "PI05_LEROBOT_CHECKPOINT",
            "PI05_LEROBOT_DEVICE",
            "PI05_LEROBOT_TOKENIZER",
        }
        defaults = {
            "PI05_LEROBOT_CHECKPOINT": str(
                ROOT / "policy" / "pi05_lerobot" / "checkpoints" / "pretrained_model"
            ),
            "PI05_LEROBOT_DEVICE": "cuda",
        }
    elif module_name in {"pi05_sim", "pi05-sim"}:
        relevant_keys = {"PI05_SIM_CHECKPOINT", "PI05_SIM_EXEC_CHUNK"}
    elif module_name in {"hamlet", "hamlet_policy"}:
        relevant_keys = {
            "HAMLET_MODEL_PATH",
            "HAMLET_DEVICE",
            "HAMLET_STRICT",
            "HAMLET_EXEC_CHUNK",
            "HAMLET_DEMO_PRIME",
            "HAMLET_MEMORY_WINDOW",
            "HAMLET_MEMORY_STRIDE",
        }
        hamlet_root = ROOT / "policy" / "HAMLET"
        bundled_checkpoint = (
            ROOT.parent / "HAMLET" / "runs" / "rmbench" / "hamlet_n1d6" / "checkpoint-60000"
        )
        defaults = {
            "HAMLET_MODEL_PATH": str(bundled_checkpoint),
            "HAMLET_DEVICE": "cuda",
            "HAMLET_STRICT": "true",
            "HAMLET_EXEC_CHUNK": "16",
            "HAMLET_DEMO_PRIME": "true",
            "HAMLET_MEMORY_WINDOW": "4",
            "HAMLET_MEMORY_STRIDE": "50",
        }
    else:
        relevant_keys = set()
    environment = {
        key: os.environ.get(key, defaults.get(key))
        for key in relevant_keys
        if key in os.environ or key in defaults
    }
    paths: dict[str, Any] = {}
    for key in _POLICY_RUNTIME_PATH_KEYS:
        value = environment.get(key)
        if not value:
            continue
        path = Path(value).expanduser()
        try:
            stat = path.stat()
        except OSError:
            paths[key] = {"path": str(path), "exists": False}
            continue
        signature: dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "is_dir": path.is_dir(),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        # Directory mtime does not change when a checkpoint file is replaced
        # in place.  Include a cheap recursive metadata digest so that weight
        # swaps invalidate resume without reading multi-GB tensor contents.
        if path.is_dir():
            entries_digest = hashlib.sha256()
            file_count = 0
            for child in sorted(path.rglob("*")):
                if child.is_file() and "__pycache__" not in child.parts:
                    try:
                        child_stat = child.stat()
                        relative = str(child.relative_to(path))
                    except OSError:
                        continue
                    entries_digest.update(relative.encode("utf-8"))
                    entries_digest.update(b"\0")
                    entries_digest.update(str(int(child_stat.st_size)).encode("ascii"))
                    entries_digest.update(b"\0")
                    entries_digest.update(str(int(child_stat.st_mtime_ns)).encode("ascii"))
                    entries_digest.update(b"\0")
                    file_count += 1
            signature["file_count"] = file_count
            signature["files_metadata_sha256"] = entries_digest.hexdigest()
        paths[key] = signature
    return {"environment": environment, "paths": paths}


def _latest_rdt_checkpoint(rdt_root: Path) -> Path:
    checkpoint_root = rdt_root / "checkpoints" / "rdt-mem-sim"
    numbered: list[tuple[int, Path]] = []
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
    return checkpoint_root


def _write_log(
    path: Path,
    results: list[dict[str, Any]],
    args,
    run_metadata: dict[str, Any] | None = None,
) -> None:
    run_metadata = run_metadata or _run_metadata(args)
    payload = {
        "format": "gitbench_eval_log_v2",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_metadata": run_metadata,
        "policy": args.policy,
        "tasks": args.tasks,
        "split": args.split,
        "episodes_per_task": args.episodes,
        "start_episode": args.start_episode,
        "max_steps_override": args.max_steps,
        "stop_on_success": args.stop_on_success,
        "success_mode": getattr(args, "success_mode", "ever"),
        "terminate_on_success": getattr(args, "terminate_on_success", False),
        "obs_mode": args.obs_mode,
        "render_mode": args.render_mode,
        "demonstration_reset_contract": (
            "policy.reset(task, episode_id, obs, info) receives "
            "info.demonstration; policy.act starts after demonstration replay"
        ),
        "save_video": bool(args.save_video),
        "video_dir": str(args.video_dir) if args.save_video else None,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _summary(results: list[dict[str, Any]], args) -> dict[str, Any]:
    task_filter = set(args.tasks)
    selected_attempts = [
        result
        for result in results
        if isinstance(result, dict)
        if result.get("task_key") in task_filter
        and _episode_key(result) is not None
    ]
    selected_results = _latest_episode_results(selected_attempts)
    by_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "episodes": 0,
            "scored_episodes": 0,
            "unscored_episodes": 0,
            "successes": 0,
            "failures": 0,
            "errors": 0,
            "infrastructure_errors": 0,
            "policy_errors": 0,
            "process_completion_sum": 0.0,
            "process_completion_count": 0,
            "milestone_counts": defaultdict(int),
        }
    )
    for result in selected_results:
        # Logs are user-editable audit artifacts.  Ignore malformed rows in
        # the aggregate rather than crashing a completed evaluation merely
        # because an interrupted write left a partial record.
        task_key = result.get("task_key")
        if not isinstance(task_key, str) or not task_key:
            continue
        row = by_task[task_key]
        row["episodes"] += 1
        outcome = _result_outcome(result)
        process_eligible = bool(result.get("scorable", True)) and outcome != "error"
        progress = result.get("progress")
        completion = (
            progress.get("process_completion")
            if isinstance(progress, dict)
            else None
        )
        if process_eligible and isinstance(completion, (int, float)) and not isinstance(completion, bool):
            row["process_completion_sum"] += float(completion)
            row["process_completion_count"] += 1
            for milestone in progress.get("milestones", []):
                if isinstance(milestone, dict) and milestone.get("achieved"):
                    milestone_id = milestone.get("id")
                    if isinstance(milestone_id, str):
                        row["milestone_counts"][milestone_id] += 1
        if result.get("scorable", True):
            if outcome == "error":
                row["errors"] += 1
                row["infrastructure_errors"] += 1
            else:
                row["scored_episodes"] += 1
                row["policy_errors"] += int(result.get("error_kind") == "policy")
                if outcome == "success":
                    row["successes"] += 1
                else:
                    row["failures"] += 1
        else:
            row["unscored_episodes"] += 1
            infrastructure_error = int(
                result.get("error_kind") == "infrastructure"
            )
            row["errors"] += infrastructure_error
            row["infrastructure_errors"] += infrastructure_error
            row["policy_errors"] += int(result.get("error_kind") == "policy")
    for row in by_task.values():
        row["provisional_success_rate"] = (
            row["successes"] / row["scored_episodes"]
            if row["scored_episodes"]
            else None
        )
        row["success_rate"] = (
            row["provisional_success_rate"]
            if row["infrastructure_errors"] == 0
            else None
        )
        process_count = row.pop("process_completion_count")
        row["mean_process_completion"] = (
            row.pop("process_completion_sum") / process_count
            if process_count
            else None
        )
        count = row.pop("scored_episodes")
        row["scored_episodes"] = count
        row["milestone_attainment_rates"] = {
            milestone_id: value / process_count
            for milestone_id, value in row.pop("milestone_counts").items()
            if process_count
        }
    scored_results = [
        result
        for result in selected_results
        if result.get("scorable", True)
        and _result_outcome(result) != "error"
    ]
    total = len(scored_results)
    successes = sum(
        1 for result in scored_results if _result_outcome(result) == "success"
    )
    infrastructure_errors = sum(
        1 for result in selected_results if _result_outcome(result) == "error"
    )
    policy_errors = sum(1 for result in selected_results if result.get("error_kind") == "policy")
    infrastructure_error_attempts = sum(
        1
        for result in selected_attempts
        if _result_outcome(result) == "error"
    )
    errors = infrastructure_errors
    failures = sum(
        1
        for result in scored_results
        if _result_outcome(result) == "failure"
    )
    completions = [
        float(result["progress"]["process_completion"])
        for result in selected_results
        if result.get("scorable", True)
        and isinstance(result.get("progress"), dict)
        and isinstance(result["progress"].get("process_completion"), (int, float))
        and not isinstance(result["progress"].get("process_completion"), bool)
        and _result_outcome(result) != "error"
    ]
    milestone_totals: dict[str, int] = defaultdict(int)
    for result in selected_results:
        progress = result.get("progress")
        if (
            not result.get("scorable", True)
            or not isinstance(progress, dict)
            or _result_outcome(result) == "error"
        ):
            continue
        for milestone in progress.get("milestones", []):
            if isinstance(milestone, dict) and milestone.get("achieved"):
                milestone_id = milestone.get("id")
                if isinstance(milestone_id, str):
                    milestone_totals[milestone_id] += 1
    failure_completions = [
        float(result["progress"]["process_completion"])
        for result in selected_results
        if result.get("scorable", True)
        and _result_outcome(result) == "failure"
        and isinstance(result.get("progress"), dict)
        and isinstance(result["progress"].get("process_completion"), (int, float))
        and not isinstance(result["progress"].get("process_completion"), bool)
    ]
    return {
        "policy": args.policy,
        "tasks": args.tasks,
        "split": args.split,
        "episodes_per_task": args.episodes,
        "max_steps_override": args.max_steps,
        "stop_on_success": args.stop_on_success,
        "success_mode": getattr(args, "success_mode", "ever"),
        "terminate_on_success": getattr(args, "terminate_on_success", False),
        "total": total,
        "attempts": len(selected_attempts),
        # ``total`` excludes runtime errors; ``unscored`` is reserved for
        # intentionally non-scorable tasks and therefore does not absorb
        # errors.
        "unscored": sum(
            1 for result in selected_results if not result.get("scorable", True)
        ),
        "successes": successes,
        "failures": failures,
        "errors": errors,
        "success_rate": (
            successes / total
            if total and infrastructure_errors == 0
            else None
        ),
        "provisional_success_rate": (
            successes / (successes + failures)
            if successes + failures
            else None
        ),
        "infrastructure_errors": infrastructure_errors,
        "infrastructure_error_attempts": infrastructure_error_attempts,
        "policy_errors": policy_errors,
        "mean_process_completion": (
            sum(completions) / len(completions) if completions else None
        ),
        "median_process_completion": (
            float(np.median(completions)) if completions else None
        ),
        "completion_at_failure": (
            sum(failure_completions) / len(failure_completions)
            if failure_completions
            else None
        ),
        "milestone_attainment_rates": {
            milestone_id: count / len(completions)
            for milestone_id, count in milestone_totals.items()
            if completions
        },
        "log_path": str(args.out),
        "results_path": str(args.out),
        "video_dir": str(args.video_dir) if args.save_video else None,
        "by_task": dict(by_task),
    }


def _latest_episode_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one canonical result per task/episode, keeping latest order."""
    latest: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        key = _episode_key(result)
        if key is None:
            continue
        latest[key] = (index, result)
    canonical = list(latest.values())
    canonical.sort(key=lambda item: item[0])
    return [result for _, result in canonical]


def _episode_key(result: dict[str, Any]) -> tuple[str, int] | None:
    """Parse the stable identity used for resume/accounting.

    Result logs are append-only and may be inspected or edited by users, so a
    malformed row must never become a phantom episode in scoring totals.
    """
    try:
        task_key = result["task_key"]
        episode_id = result["episode_id"]
        if not isinstance(task_key, str) or not task_key:
            return None
        if isinstance(episode_id, bool):
            return None
        return task_key, int(episode_id)
    except (KeyError, TypeError, ValueError):
        return None


def _result_outcome(result: dict[str, Any]) -> str:
    """Normalize current and legacy result rows to one scoring outcome."""
    error_kind = result.get("error_kind")
    if error_kind == "infrastructure":
        return "error"
    if (
        error_kind == "policy"
        or result.get("policy_load_error")
        or result.get("policy_reset_error")
        or result.get("policy_act_error")
    ):
        return "failure"
    outcome = result.get("outcome")
    if outcome in {"success", "failure", "error"}:
        return outcome
    if result.get("error"):
        return "error"
    return "success" if bool(result.get("success", False)) else "failure"


def _outcome(success: bool, error: str | None) -> str:
    if error == "infrastructure":
        return "error"
    if error == "policy":
        return "failure"
    return "success" if success else "failure"


def _error_kind(
    reset_error: str | None,
    policy_load_error: str | None,
    policy_reset_error: str | None,
    policy_act_error: str | None,
    error: str | None,
) -> str | None:
    if policy_load_error or policy_reset_error or policy_act_error:
        return "policy"
    if reset_error or error:
        return "infrastructure"
    return None


def _render_frame(env: GITBenchEnv) -> tuple[np.ndarray | None, str | None]:
    try:
        frame = env.render()
        frame = _extract_frame(frame)
        return _to_uint8_rgb(frame), None
    except Exception as exc:
        return None, repr(exc)


class _EpisodeVideoRecorder:
    """Stream one episode video without retaining large render frames in RAM."""

    def __init__(self, video_dir: Path, task_key: str, episode_id: int, fps: int) -> None:
        self.task_key = task_key
        self.episode_id = episode_id
        self.fps = fps
        self.task_video_dir = video_dir / task_key
        self.temp_path = self.task_video_dir / f".episode{episode_id}.recording.mp4"
        self.writer = None
        self.error: str | None = None
        self.frame_count = 0
        self.demonstration_frame_count = 0
        self.interaction_frame_count = 0

    def capture(self, env: GITBenchEnv, *, is_demonstration: bool) -> None:
        if self.error is not None:
            return
        frame, error = _render_frame(env)
        if error is not None:
            self.error = error
            return
        if frame is not None:
            self.append(frame, is_demonstration=is_demonstration)

    def append(self, frame: np.ndarray, *, is_demonstration: bool) -> None:
        if self.error is not None:
            return
        try:
            output = _add_demonstration_border(frame) if is_demonstration else frame
            if self.writer is None:
                self.task_video_dir.mkdir(parents=True, exist_ok=True)
                self.writer = imageio.get_writer(self.temp_path, fps=self.fps)
            self.writer.append_data(_to_uint8_rgb(output))
            self.frame_count += 1
            if is_demonstration:
                self.demonstration_frame_count += 1
            else:
                self.interaction_frame_count += 1
        except Exception as exc:
            self.error = repr(exc)

    def finish(self, outcome: str) -> tuple[Path | None, str | None]:
        try:
            if self.writer is None:
                return None, self.error or "No video frames were produced."
            self.writer.close()
            self.writer = None
            path = self.task_video_dir / f"episode{self.episode_id}_{outcome}.mp4"
            self.temp_path.replace(path)
            return path, self.error
        except Exception as exc:
            return None, self.error or repr(exc)


def _video_log_fields(recorder: _EpisodeVideoRecorder | None) -> dict[str, int | None]:
    if recorder is None:
        return {
            "video_frame_count": None,
            "video_demonstration_frame_count": None,
            "video_interaction_frame_count": None,
        }
    return {
        "video_frame_count": recorder.frame_count,
        "video_demonstration_frame_count": recorder.demonstration_frame_count,
        "video_interaction_frame_count": recorder.interaction_frame_count,
    }


def _add_demonstration_border(frame: np.ndarray) -> np.ndarray:
    """Mark video-only demonstration frames without changing model observations."""
    image = _to_uint8_rgb(frame).copy()
    height, width = image.shape[:2]
    border = max(4, min(16, min(height, width) // 40))
    red = np.asarray([235, 32, 32], dtype=np.uint8)
    image[:border, :, :] = red
    image[-border:, :, :] = red
    image[:, :border, :] = red
    image[:, -border:, :] = red
    return image


def _write_video(
    frames: list[np.ndarray],
    video_dir: Path,
    task_key: str,
    episode_id: int,
    fps: int,
    prior_error: str | None,
    outcome: str,
) -> tuple[Path | None, str | None]:
    task_video_dir = video_dir / task_key
    task_video_dir.mkdir(parents=True, exist_ok=True)
    path = task_video_dir / f"episode{episode_id}_{outcome}.mp4"
    try:
        imageio.mimsave(path, frames, fps=fps)
        return path, prior_error
    except Exception as exc:
        return None, repr(exc)


def _video_outcome(outcome: str) -> str:
    return "fail" if outcome == "failure" else outcome


def _to_uint8_rgb(frame: Any) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    if hasattr(frame, "numpy"):
        frame = frame.numpy()
    arr = np.asarray(frame)
    if arr.ndim < 2:
        raise ValueError(f"Render result is not image-like: shape={arr.shape}")
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] < 3:
        raise ValueError(f"Render result does not have RGB channels: shape={arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0 if arr.max(initial=0) <= 1.0 else np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _extract_frame(frame: Any) -> Any:
    """Find an image-like array inside ManiSkill render outputs."""
    if frame is None:
        raise ValueError("Render result is None")

    if isinstance(frame, dict):
        preferred = (
            "render_camera",
            "human_cam",
            "human_camera",
            "base_camera",
            "rgb",
            "rgba",
            "Color",
            "color",
            "image",
        )
        for key in preferred:
            if key in frame:
                try:
                    return _extract_frame(frame[key])
                except Exception:
                    pass
        for value in frame.values():
            try:
                return _extract_frame(value)
            except Exception:
                pass
        raise ValueError(f"No image-like value in render dict keys={list(frame.keys())}")

    if isinstance(frame, (list, tuple)):
        if not frame:
            raise ValueError("Empty render result")
        for value in frame:
            try:
                return _extract_frame(value)
            except Exception:
                pass
        raise ValueError(f"No image-like value in render sequence length={len(frame)}")

    shape = getattr(frame, "shape", None)
    if shape is not None:
        if len(shape) >= 2:
            return frame
        raise ValueError(f"Render result is not image-like: shape={tuple(shape)}")

    arr = np.asarray(frame)
    if arr.ndim >= 2:
        return frame
    raise ValueError(f"Render result is not image-like: shape={arr.shape}")


def _diagnostic_frame(task_key: str, episode_id: int, message: str) -> np.ndarray:
    frame = np.zeros((368, 640, 3), dtype=np.uint8)
    frame[:, :] = [32, 32, 36]
    frame[:72, :] = [120, 36, 36]
    frame[72:76, :] = [220, 180, 80]
    try:
        from PIL import Image, ImageDraw

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        lines = [
            "GITBench diagnostic video",
            f"task={task_key} episode={episode_id}",
            "No simulator frame was rendered.",
            message[:110],
        ]
        y = 22
        for line in lines:
            draw.text((24, y), line, fill=(255, 255, 255))
            y += 34
        return np.asarray(image)
    except Exception:
        # Keep a visible nonblank frame even if Pillow is unavailable.
        for idx, value in enumerate(task_key.encode("utf-8")[:24]):
            x0 = 24 + idx * 16
            frame[112:260, x0 : x0 + 10] = [value, 180, 255 - value]
        return frame


if __name__ == "__main__":
    main()
