"""Process-completion milestone configuration and validation.

Progress is a supplementary evaluator signal.  It is deliberately kept
separate from the benchmark's binary success/failure contract.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping


PROGRESS_VERSION = "gitbench_progress_v1"
SUPPORTED_PROGRESS_RULES = {
    "object_grasped",
    "object_displaced_from_reset",
    "object_lifted_from_reset",
    "object_in_rectangle",
    "object_near_point",
    "object_in_circle",
    "object_near_initial_position",
    "lid_open",
    "plug_inserted",
}


def _milestone(
    milestone_id: str,
    weight: float,
    rule: str,
    params: Mapping[str, Any],
    *,
    requires: list[str] | None = None,
) -> dict[str, Any]:
    value = {
        "id": milestone_id,
        "weight": float(weight),
        "rule": rule,
        "params": deepcopy(dict(params)),
    }
    if requires:
        value["requires"] = list(requires)
    return value


def default_progress(task_key: str | None, evaluator: Mapping[str, Any]) -> dict[str, Any]:
    """Build semantic milestones from an existing final evaluator rule.

    This keeps the checked-in v3 manifest backwards compatible: explicit
    ``evaluator.progress`` blocks can override these defaults per episode.
    """
    rule = str(evaluator.get("rule", ""))
    params = dict(evaluator)
    params.pop("failure_rules", None)
    milestones: list[dict[str, Any]] = []

    if rule == "objects_in_pad":
        refs = list(params.get("object_refs", []))
        if refs:
            weight = 1.0 / len(refs)
            for ref in refs:
                milestones.append(
                    _milestone(
                        f"{ref}_in_goal",
                        weight,
                        "object_in_rectangle",
                        {**params, "object_ref": ref},
                    )
                )
    elif rule == "lid_open":
        milestones.append(_milestone("target_lid_open", 1.0, "lid_open", params))
    elif rule == "plug_inserted":
        milestones.extend(
            [
                _milestone(
                    "target_plug_moved",
                    0.35,
                    "object_displaced_from_reset",
                    {"object_ref": params.get("object_ref"), "threshold": 0.03, "metric": "xyz"},
                ),
                _milestone(
                    "target_plug_inserted",
                    0.65,
                    "plug_inserted",
                    params,
                    requires=["target_plug_moved"],
                ),
            ]
        )
    elif rule == "object_near_initial_position":
        # Returning to the initial pose should not score unless the object
        # first left it; this prevents a reset-state false positive.
        ref = params.get("object_ref")
        moved = _milestone(
            "target_left_initial_position",
            0.35,
            "object_displaced_from_reset",
            {"object_ref": ref, "threshold": 0.03, "metric": "xyz"},
        )
        returned = _milestone(
            "target_returned",
            0.65,
            "object_near_initial_position",
            params,
            requires=["target_left_initial_position"],
        )
        milestones.extend([moved, returned])
    elif rule in {"object_in_rectangle", "object_near_point", "object_in_circle"}:
        ref = params.get("object_ref")
        milestones.extend(
            [
                _milestone(
                    "target_grasped",
                    0.20,
                    "object_grasped",
                    {"object_ref": ref, "stable_steps": 2},
                ),
                _milestone(
                    "target_moved",
                    0.20,
                    "object_displaced_from_reset",
                    {"object_ref": ref, "threshold": 0.03, "metric": "xyz"},
                    requires=["target_grasped"],
                ),
                _milestone(
                    "target_in_goal",
                    0.60,
                    rule,
                    params,
                    requires=["target_moved"],
                ),
            ]
        )

    # A malformed or unsupported task should not make evaluation crash.  It
    # receives an empty, explicitly versioned progress contract instead.
    return {"version": PROGRESS_VERSION, "aggregation": "weighted_sum", "milestones": milestones}


def normalize_progress(
    progress: Mapping[str, Any] | None,
    evaluator: Mapping[str, Any],
    *,
    task_key: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize an explicit or evaluator-derived progress block."""
    source = default_progress(task_key, evaluator) if progress is None else deepcopy(dict(progress))
    if source.get("version", PROGRESS_VERSION) != PROGRESS_VERSION:
        raise ValueError(f"Unsupported progress version {source.get('version')!r}")
    source["version"] = PROGRESS_VERSION
    if source.get("aggregation", "weighted_sum") != "weighted_sum":
        raise ValueError("progress aggregation must be 'weighted_sum'")
    raw = source.get("milestones", [])
    if not isinstance(raw, list):
        raise ValueError("progress.milestones must be a list")
    milestones: list[dict[str, Any]] = []
    ids: set[str] = set()
    weights = 0.0
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"progress milestone {index} must be a mapping")
        milestone_id = item.get("id")
        rule = item.get("rule")
        weight = item.get("weight")
        params = item.get("params", {})
        if not isinstance(milestone_id, str) or not milestone_id:
            raise ValueError(f"progress milestone {index} id must be a non-empty string")
        if milestone_id in ids:
            raise ValueError(f"duplicate progress milestone id {milestone_id!r}")
        ids.add(milestone_id)
        if rule not in SUPPORTED_PROGRESS_RULES:
            raise ValueError(f"unsupported progress rule {rule!r}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError(f"progress milestone {milestone_id!r} weight must be positive and finite")
        if not isinstance(params, Mapping):
            raise ValueError(f"progress milestone {milestone_id!r} params must be a mapping")
        requires = item.get("requires", [])
        if not isinstance(requires, list) or any(not isinstance(dep, str) or not dep for dep in requires):
            raise ValueError(f"progress milestone {milestone_id!r} requires must be a list of strings")
        stable_steps = params.get("stable_steps", 1)
        if isinstance(stable_steps, bool) or not isinstance(stable_steps, int) or stable_steps < 1:
            raise ValueError(f"progress milestone {milestone_id!r} stable_steps must be a positive integer")
        normalized = {
            "id": milestone_id,
            "weight": float(weight),
            "rule": str(rule),
            "params": deepcopy(dict(params)),
            "requires": list(requires),
        }
        milestones.append(normalized)
        weights += float(weight)
    if milestones and not math.isclose(weights, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"progress milestone weights must sum to 1.0, got {weights}")
    unknown = {dep for item in milestones for dep in item["requires"] if dep not in ids}
    if unknown:
        raise ValueError(f"progress requires unknown milestones {sorted(unknown)}")
    # Dependencies define the order in which milestones may be awarded.  A
    # cycle can never make progress and would otherwise fail silently at
    # runtime, so reject it while loading the manifest.
    dependencies = {item["id"]: set(item["requires"]) for item in milestones}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(milestone_id: str) -> None:
        if milestone_id in visiting:
            raise ValueError(f"progress milestone dependency cycle at {milestone_id!r}")
        if milestone_id in visited:
            return
        visiting.add(milestone_id)
        for dependency in dependencies[milestone_id]:
            visit(dependency)
        visiting.remove(milestone_id)
        visited.add(milestone_id)

    for milestone_id in dependencies:
        visit(milestone_id)
    source["milestones"] = milestones
    return source
