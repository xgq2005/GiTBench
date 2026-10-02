"""Canonical GITBench episode schema and manifest access."""

from __future__ import annotations

from copy import deepcopy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
import math

from .demonstrations import HISTORY_TIMING_SOURCE
from .paths import repo_root
from .progress import normalize_progress


Episode = dict[str, Any]

EPISODE_MANIFEST_FORMAT = "gitbench_episode_manifest_v3"
REQUIRED_EPISODE_FIELDS = {
    "episode_id",
    "source",
    "split",
    "public",
    "reset",
    "evaluator",
}

RULE_REQUIRED_PARAMS = {
    "objects_in_pad": {"object_refs", "center", "half_extents", "metric"},
    "object_in_rectangle": {"object_ref", "center", "half_extents", "metric"},
    "object_near_point": {"object_ref", "target", "metric", "threshold"},
    "lid_open": {"object_ref"},
    "plug_inserted": {
        "object_ref",
        "socket_ref",
        "lateral_threshold",
        "min_insert_depth",
    },
    "object_in_circle": {"object_ref", "center", "radius", "metric"},
    "object_near_initial_position": {"object_ref", "metric", "threshold"},
}

FAILURE_RULE_REQUIRED_PARAMS = {
    "wrong_object_interaction": {"object_refs", "lift_threshold", "move_threshold"},
    "wrong_object_in_rectangle": {
        "object_refs",
        "center",
        "half_extents",
        "metric",
    },
    "wrong_object_near_point": {
        "object_refs",
        "target",
        "metric",
        "threshold",
    },
    "wrong_object_in_circle": {"object_refs", "center", "radius", "metric"},
    "wrong_lid_open": {"object_refs"},
    "wrong_plug_interaction": {
        "object_refs",
        "lift_threshold",
        "move_threshold",
    },
    "wrong_plug_in_socket": {
        "object_refs",
        "socket_ref",
        "lateral_threshold",
        "min_insert_depth",
    },
}

LAYOUT_REF_DOMAINS = {
    "bio2": {f"C{index}" for index in range(1, 6)},
    "ind2": {f"B{index}" for index in range(1, 6)},
    "ind3": {f"S{index}" for index in range(1, 7)},
    "ind5": {f"P{index}" for index in range(1, 6)},
}


def manifest_path() -> Path:
    return repo_root() / "task_config" / "test_manifest.json"


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    path = manifest_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != EPISODE_MANIFEST_FORMAT:
        raise ValueError(f"Unsupported test manifest format in {path}")
    if not isinstance(payload.get("tasks"), dict):
        raise ValueError(f"Missing tasks mapping in {path}")
    _validate_manifest(payload)
    return payload


def _validate_manifest(payload: Mapping[str, Any]) -> None:
    """Validate every manifest row before an evaluator can consume it.

    The old loader only checked top-level JSON shape; malformed evaluator
    references then silently resolved to ``None`` in the simulator.  Keep this
    pass simulator-independent and strict about schema/value invariants.
    """
    for task_key, rows in payload["tasks"].items():
        if not isinstance(rows, list):
            raise ValueError(f"Task {task_key!r} episodes must be a list")
        seen_ids: set[int] = set()
        for row_index, episode in enumerate(rows):
            if not isinstance(episode, Mapping):
                raise ValueError(f"Task {task_key!r} episode {row_index} must be a mapping")
            missing = REQUIRED_EPISODE_FIELDS - set(episode)
            if missing:
                raise ValueError(
                    f"Task {task_key!r} episode {row_index} missing fields {sorted(missing)}"
                )
            try:
                episode_id = int(episode["episode_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Task {task_key!r} episode id must be an integer") from exc
            if episode_id in seen_ids:
                raise ValueError(f"Task {task_key!r} has duplicate episode_id {episode_id}")
            seen_ids.add(episode_id)
            if not isinstance(episode["split"], str) or not episode["split"]:
                raise ValueError(f"Task {task_key!r} episode {episode_id} has invalid split")
            for field in ("source", "public", "reset"):
                if not isinstance(episode[field], Mapping):
                    raise ValueError(
                        f"Task {task_key!r} episode {episode_id} field {field!r} must be a mapping"
                    )
            source = episode["source"]
            if not isinstance(source.get("family"), str) or not source.get("family"):
                raise ValueError(
                    f"Task {task_key!r} episode {episode_id} source.family must be a non-empty string"
                )
            if not isinstance(source.get("record_id"), int) or isinstance(
                source.get("record_id"), bool
            ):
                raise ValueError(
                    f"Task {task_key!r} episode {episode_id} source.record_id must be an integer"
                )
            # Reuse the detailed evaluator checks below, without requiring a
            # simulator or task registry to be importable.
            evaluator_config(episode, task_key=task_key)


def _validate_metric(metric: Any, context: str) -> str:
    if metric not in {"xy", "xyz"}:
        raise ValueError(f"{context} metric must be 'xy' or 'xyz', got {metric!r}")
    return str(metric)


def _validate_vector(value: Any, dimensions: int, context: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) < dimensions:
        raise ValueError(f"{context} must contain at least {dimensions} numeric values")
    for item in value[:dimensions]:
        # ``bool`` is an ``int`` subclass, but accepting it here turns a
        # malformed manifest such as ``center: [true, 0]`` into a valid
        # geometry rule.  Numeric evaluator parameters must be real numbers.
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{context} must contain finite numeric values")


def _validate_nonnegative_vector(value: Any, dimensions: int, context: str) -> None:
    _validate_vector(value, dimensions, context)
    if any(float(item) < 0 for item in value[:dimensions]):
        raise ValueError(f"{context} must contain non-negative values")


def _validate_nonnegative(value: Any, context: str, *, strictly_positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context} must be a finite numeric value")
    if strictly_positive and float(value) <= 0:
        raise ValueError(f"{context} must be > 0")
    if not strictly_positive and float(value) < 0:
        raise ValueError(f"{context} must be >= 0")


def available_splits(task_key: str) -> tuple[str, ...]:
    rows = load_manifest()["tasks"].get(task_key, [])
    return tuple(sorted({str(row["split"]) for row in rows}))


def get_episodes(
    task_key: str,
    *,
    split: str = "test",
    limit: int | None = None,
    start: int = 0,
) -> list[Episode]:
    rows = load_manifest()["tasks"].get(task_key)
    if rows is None:
        raise KeyError(f"Task '{task_key}' is not present in the test manifest")
    selected = [
        row
        for row in rows
        if row.get("split") == split
        or (split == "test" and str(row.get("split", "")).startswith("test-"))
    ]
    selected.sort(key=lambda row: int(row["episode_id"]))
    if start:
        selected = selected[start:]
    if limit is not None:
        selected = selected[:limit]
    return deepcopy(selected)


def get_episode(task_key: str, episode_id: int, *, split: str = "test") -> Episode:
    for row in get_episodes(task_key, split=split):
        if int(row["episode_id"]) == int(episode_id):
            return row
    known = [row["episode_id"] for row in get_episodes(task_key, split=split)]
    raise KeyError(
        f"No {split!r} episode {episode_id} for task {task_key!r}. "
        f"Available ids: {known[:5]}{'...' if len(known) > 5 else ''}"
    )


def public_episode_info(episode: Mapping[str, Any]) -> dict[str, Any]:
    public = dict(episode.get("public", {}))
    history = public.get("history", {})
    if not isinstance(history, Mapping):
        history = {}
    return {
        "episode_id": int(episode["episode_id"]),
        "split": str(episode["split"]),
        "instruction": str(public.get("instruction", "")),
        "history_events": deepcopy(list(history.get("events", []))),
        "history_timing_source": str(
            history.get("timing_source", HISTORY_TIMING_SOURCE)
        ),
    }


def episode_reset_config(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a canonical reset block for the task-specific environment adapter."""
    reset = episode.get("reset", {})
    if not isinstance(reset, Mapping):
        raise ValueError(f"Episode {episode.get('episode_id')} reset must be a mapping")
    params = reset.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError(f"Episode {episode.get('episode_id')} reset.params must be a mapping")
    return {
        "episode_id": int(episode["episode_id"]),
        "layout_id": int(reset.get("layout_id", 0)),
        **deepcopy(dict(params)),
    }


def episode_seed(episode: Mapping[str, Any]) -> int:
    reset = episode.get("reset", {})
    if not isinstance(reset, Mapping):
        return int(episode["episode_id"])
    return int(reset.get("seed", episode["episode_id"]))


def _active_layout_refs(task_key: str | None, episode: Mapping[str, Any]) -> set[str] | None:
    """Return object labels instantiated by an episode's explicit layout.

    LPX layouts intentionally use a fixed-width evaluator template (for
    example C1..C5) even when an episode instantiates fewer objects.  Resolve
    that template against the active layout once, instead of letting each
    simulator adapter silently handle an absent object differently.
    """
    if task_key not in {"bio2", "ind2", "ind3", "ind5"}:
        return None
    reset = episode.get("reset", {})
    params = reset.get("params", {}) if isinstance(reset, Mapping) else {}
    layout = params.get("layout_config", {}) if isinstance(params, Mapping) else {}
    if not isinstance(layout, Mapping):
        return set()
    if task_key == "bio2":
        values = layout.get("containers", [])
        return set(values) if isinstance(values, list) else set()
    if task_key == "ind2":
        try:
            count = int(layout.get("num_boxes", 0))
        except (TypeError, ValueError):
            count = 0
        return {f"B{index}" for index in range(1, max(0, count) + 1)}
    if task_key == "ind3":
        values = layout.get("sockets", [])
        return set(values) if isinstance(values, list) else set()
    values = layout.get("parts", [])
    return set(values) if isinstance(values, list) else set()


def _validate_active_ref(
    ref: Any,
    active_refs: set[str] | None,
    context: str,
    *,
    required: bool = True,
) -> None:
    if active_refs is None or ref is None:
        return
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"{context} must be a non-empty string")
    if required and ref not in active_refs:
        raise ValueError(f"{context} {ref!r} is not present in the episode layout")


def _validate_layout_domain_ref(ref: str, task_key: str | None, context: str) -> None:
    domain = LAYOUT_REF_DOMAINS.get(task_key)
    if domain is not None and ref not in domain:
        raise ValueError(f"{context} {ref!r} is not a valid {task_key} object reference")


def evaluator_config(
    episode: Mapping[str, Any], *, task_key: str | None = None
) -> dict[str, Any]:
    """Flatten evaluator rule parameters for internal benchmark use only."""
    evaluator = episode.get("evaluator", {})
    if not isinstance(evaluator, Mapping):
        raise ValueError(f"Episode {episode.get('episode_id')} evaluator must be a mapping")
    params = evaluator.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError(
            f"Episode {episode.get('episode_id')} evaluator.params must be a mapping"
        )
    rule = evaluator.get("rule")
    if not isinstance(rule, str) or not rule:
        raise ValueError(f"Episode {episode.get('episode_id')} has no evaluator rule")
    if rule not in RULE_REQUIRED_PARAMS:
        raise ValueError(f"Episode {episode.get('episode_id')} uses unknown rule {rule!r}")
    missing = RULE_REQUIRED_PARAMS[rule] - set(params)
    if missing:
        raise ValueError(
            f"Episode {episode.get('episode_id')} evaluator {rule!r} is missing {sorted(missing)}"
        )
    metric = params.get("metric")
    if metric is not None:
        _validate_metric(metric, f"Episode {episode.get('episode_id')} evaluator")
    vector_specs = {
        "center": 3 if metric == "xyz" else 2,
        "half_extents": 3 if metric == "xyz" else 2,
        "target": 3 if metric == "xyz" else 2,
    }
    for key, dimensions in vector_specs.items():
        if key in params:
            validator = _validate_nonnegative_vector if key == "half_extents" else _validate_vector
            validator(params[key], dimensions, f"evaluator.{key}")
    for key in ("threshold", "radius", "lateral_threshold", "min_insert_depth"):
        if key in params:
            _validate_nonnegative(params[key], f"evaluator.{key}")
    active_refs = _active_layout_refs(task_key, episode)
    refs = params.get("object_refs")
    if "object_refs" in params and (
        not isinstance(refs, list)
        or not refs
        or any(not isinstance(ref, str) or not ref for ref in refs)
        or len(set(refs)) != len(refs)
    ):
        raise ValueError(
            f"Episode {episode.get('episode_id')} evaluator.object_refs must be a non-empty list of unique strings"
        )
    if isinstance(refs, list):
        for ref in refs:
            _validate_layout_domain_ref(
                ref,
                task_key,
                f"Episode {episode.get('episode_id')} evaluator.object_refs",
            )
            _validate_active_ref(
                ref,
                active_refs,
                f"Episode {episode.get('episode_id')} evaluator.object_refs",
            )
    for key in ("object_ref", "socket_ref"):
        if key in params and (not isinstance(params[key], str) or not params[key]):
            raise ValueError(
                f"Episode {episode.get('episode_id')} evaluator.{key} must be a non-empty string"
            )
        if key in params:
            _validate_layout_domain_ref(
                params[key],
                task_key,
                f"Episode {episode.get('episode_id')} evaluator.{key}",
            )
            _validate_active_ref(
                params[key],
                active_refs,
                f"Episode {episode.get('episode_id')} evaluator.{key}",
            )
    failure_rules = evaluator.get("failure_rules", [])
    if not isinstance(failure_rules, list):
        raise ValueError(
            f"Episode {episode.get('episode_id')} evaluator.failure_rules must be a list"
        )
    normalized_failure_rules: list[dict[str, Any]] = []
    for index, failure_rule in enumerate(failure_rules):
        if not isinstance(failure_rule, Mapping):
            raise ValueError(
                f"Episode {episode.get('episode_id')} failure rule {index} must be a mapping"
            )
        failure_rule_name = failure_rule.get("rule")
        failure_params = failure_rule.get("params", {})
        if failure_rule_name not in FAILURE_RULE_REQUIRED_PARAMS:
            raise ValueError(
                f"Episode {episode.get('episode_id')} uses unknown failure rule "
                f"{failure_rule_name!r}"
            )
        if not isinstance(failure_params, Mapping):
            raise ValueError(
                f"Episode {episode.get('episode_id')} failure rule {failure_rule_name!r} "
                "params must be a mapping"
            )
        missing_failure_params = (
            FAILURE_RULE_REQUIRED_PARAMS[failure_rule_name] - set(failure_params)
        )
        if missing_failure_params:
            raise ValueError(
                f"Episode {episode.get('episode_id')} failure rule {failure_rule_name!r} "
                f"is missing {sorted(missing_failure_params)}"
            )
        failure_metric = failure_params.get("metric")
        if failure_metric is not None:
            _validate_metric(
                failure_metric,
                f"Episode {episode.get('episode_id')} failure rule {failure_rule_name}",
            )
        failure_dimensions = 3 if failure_metric == "xyz" else 2
        for key in ("center", "half_extents", "target"):
            if key in failure_params:
                validator = (
                    _validate_nonnegative_vector if key == "half_extents" else _validate_vector
                )
                validator(
                    failure_params[key],
                    failure_dimensions,
                    f"failure rule {failure_rule_name}.{key}",
                )
        for key in (
            "threshold",
            "radius",
            "lateral_threshold",
            "min_insert_depth",
            "lift_threshold",
            "move_threshold",
        ):
            if key in failure_params:
                _validate_nonnegative(
                    failure_params[key],
                    f"failure rule {failure_rule_name}.{key}",
                )
        refs = failure_params.get("object_refs")
        if "object_refs" in failure_params and (
            not isinstance(refs, list)
            or not refs
            or any(not isinstance(ref, str) or not ref for ref in refs)
            or len(set(refs)) != len(refs)
        ):
            raise ValueError(
                f"Episode {episode.get('episode_id')} failure rule {failure_rule_name}.object_refs must be a non-empty list of unique strings"
            )
        if isinstance(refs, list) and active_refs is not None:
            for ref in refs:
                _validate_layout_domain_ref(
                    ref,
                    task_key,
                    f"Episode {episode.get('episode_id')} failure rule {failure_rule_name}.object_refs",
                )
            # The checked-in LPX manifest uses a max-capacity template.  Only
            # objects actually instantiated by this episode are meaningful
            # failure candidates; normalize the template to that active set.
            refs = [ref for ref in refs if ref in active_refs]
            if not refs:
                raise ValueError(
                    f"Episode {episode.get('episode_id')} failure rule {failure_rule_name} has no active object references"
                )
        for key in ("object_ref", "socket_ref"):
            if key in failure_params:
                _validate_layout_domain_ref(
                    failure_params[key],
                    task_key,
                    f"Episode {episode.get('episode_id')} failure rule {failure_rule_name}.{key}",
                )
                _validate_active_ref(
                    failure_params[key],
                    active_refs,
                    f"Episode {episode.get('episode_id')} failure rule {failure_rule_name}.{key}",
                )
        normalized_failure_rules.append(
            {
                "rule": failure_rule_name,
                **deepcopy(dict(failure_params)),
                **({"object_refs": refs} if isinstance(refs, list) else {}),
            }
        )
    progress = normalize_progress(
        evaluator.get("progress"),
        {"rule": rule, **dict(params)},
        task_key=task_key,
    )
    return {
        "rule": rule,
        **deepcopy(dict(params)),
        "failure_rules": normalized_failure_rules,
        "progress": progress,
    }
