"""Task registry loaded from the canonical project-local task catalog."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .assets import patch_task_assets
from .paths import ensure_maniskill_on_path, repo_root


EpisodeConfig = Dict[str, Any]


@dataclass(frozen=True)
class TaskSpec:
    key: str
    env_id: str
    gym_env_id: str
    domain: str
    source_family: str
    env_imports: tuple[str, ...]
    default_episode: Mapping[str, Any]
    default_evaluator: Mapping[str, Any]
    evaluator_rule: str
    evaluator_rules: tuple[str, ...]
    success_rule: str
    max_steps: int
    scorable: bool = True

    def episode_config(self, overrides: Mapping[str, Any] | None = None) -> EpisodeConfig:
        cfg = dict(self.default_episode)
        if overrides:
            cfg.update({k: v for k, v in overrides.items() if v is not None})
        return cfg

def _catalog_path() -> Path:
    return repo_root() / "task_config" / "task_catalog.json"


def _load_tasks() -> dict[str, TaskSpec]:
    path = _catalog_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "gitbench_task_catalog_v1":
        raise ValueError(f"Unsupported task catalog format in {path}")

    tasks: dict[str, TaskSpec] = {}
    for row in payload.get("tasks", []):
        environment = row["environment"]
        default_reset = row["default_reset"]
        evaluator = row["evaluator"]
        evaluator_rules = tuple(str(rule) for rule in evaluator["rules"])
        default_episode = {
            "episode_id": int(default_reset.get("seed", 0)),
            "layout_id": int(default_reset.get("layout_id", 0)),
            **dict(default_reset.get("params", {})),
        }
        spec = TaskSpec(
            key=str(row["id"]),
            env_id=str(environment["gym_id"]),
            gym_env_id=str(environment["gym_id"]),
            domain=str(row["domain"]),
            source_family=str(row["source"]["family"]),
            env_imports=(str(environment["module"]),),
            default_episode=default_episode,
            default_evaluator={
                "rule": evaluator_rules[0],
                **dict(evaluator.get("default_params", {})),
            },
            evaluator_rule=evaluator_rules[0],
            evaluator_rules=evaluator_rules,
            success_rule=str(evaluator["description"]),
            max_steps=int(row.get("max_steps", 50)),
        )
        if spec.key in tasks:
            raise ValueError(f"Duplicate task id {spec.key!r} in {path}")
        tasks[spec.key] = spec
    if not tasks:
        raise ValueError(f"No tasks defined in {path}")
    return tasks


TASKS = _load_tasks()

ALIASES = {spec.env_id: key for key, spec in TASKS.items()}
ALIASES.update({key: key for key in TASKS})


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[ALIASES[name]]
    except KeyError as exc:
        valid = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task '{name}'. Valid tasks: {valid}") from exc


def list_tasks() -> Iterable[TaskSpec]:
    return TASKS.values()


def import_all_task_envs() -> None:
    ensure_maniskill_on_path()
    for spec in TASKS.values():
        for module in spec.env_imports:
            import_module(module)
    patch_task_assets()
