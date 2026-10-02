"""History demonstrations for GITBench.

Most task families use scene-only replay. ``bio5``, ``ind5``, and ``house5``
execute their demonstrations through dual-Piper control so robot motion,
grasping, and object trajectories remain physically coherent.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import time
from typing import Any, Callable, Mapping

import numpy as np
import sapien
import torch


PRIMARY_CAMERA_ORDER = ("base_camera", "human_cam", "sensor_cam")
HISTORY_TIMING_SOURCE = "source_defined"
# v4 keeps the source trajectory and physics timing intact, but stores a
# strided subset of the rendered demonstration frames.  The long bimanual
# demonstrations are sampled more aggressively to avoid overwhelming policy
# context windows.
DEMO_FRAME_STRIDES = {
    "bio5": 8,
    "ind5": 8,
    "house5": 8,
}
DEFAULT_DEMO_FRAME_STRIDE = 4
PLUG_SOURCE_QUATERNION = np.asarray([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float32)
BIO4_ROD_HOME_POSITION = np.asarray([10.0, 0.2, 0.15], dtype=np.float32)
BIO4_ROD_HIDE_HEIGHT = 1.5
BIO4_DROP_QUATERNION = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
BIO4_HOME_QUATERNION = np.asarray([0.70710678, 0.0, 0.70710678, 0.0], dtype=np.float32)
BIO4_VERTICAL_ROD_QUATERNION = np.asarray(
    [0.0, 0.70710678, 0.0, 0.70710678], dtype=np.float32
)
BIO5_HANDOFF_POSITION = np.asarray([-0.05, -0.065, 0.12], dtype=np.float32)
BIO5_D_RACK_POSITIONS = (
    np.asarray([-0.29875, 0.2, 0.1], dtype=np.float32),
    np.asarray([-0.345, 0.2, 0.1], dtype=np.float32),
    np.asarray([-0.39125, 0.2, 0.1], dtype=np.float32),
)
BIO5_DD_RACK_POSITIONS = (
    np.asarray([-0.29875, -0.2, 0.1], dtype=np.float32),
    np.asarray([-0.345, -0.2, 0.1], dtype=np.float32),
    np.asarray([-0.39125, -0.2, 0.1], dtype=np.float32),
)
BIO5_GRASP_HEIGHT = 0.038
BIO5_LIFT_HEIGHT = 0.16
BIO5_GRASP_QUATERNION = np.asarray(
    [0.6830127, -0.1830127, 0.6830127, -0.1830127], dtype=np.float32
)
IND5_HANDOFF_POSITION = np.asarray([-0.5, 0.05, 0.2], dtype=np.float32)
IND5_PACKING_PLACE_POSITIONS = (
    np.asarray([-0.15, -0.05, 0.025], dtype=np.float32),
    np.asarray([-0.3, -0.1, 0.025], dtype=np.float32),
    np.asarray([-0.3, 0.0, 0.025], dtype=np.float32),
)
IND5_HANDOFF_APPROACHING_LEFT = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
IND5_HANDOFF_APPROACHING_RIGHT = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
IND5_HANDOFF_CLOSING = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
IND5_SOURCE_APPROACHING = np.asarray([0.707, 0.0, -0.707], dtype=np.float32)
IND5_SOURCE_CLOSING_LEFT = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
HOUSE5_TABLE_TARGETS = (
    np.asarray([-0.35, 0.0, 0.05], dtype=np.float32),
    np.asarray([-0.4, -0.03, 0.05], dtype=np.float32),
    np.asarray([-0.375, 0.03, 0.05], dtype=np.float32),
)


EVENT_SCHEMAS: dict[str, dict[str, dict[str, set[str]]]] = {
    "bio2": {
        "pour": {"required": {"source", "target"}, "allowed": {"source", "target"}},
        "swap_containers": {
            "required": {"object1", "object2"},
            "allowed": {"object1", "object2"},
        },
        "fake_pour": {
            "required": {"source", "target"},
            "allowed": {"source", "target"},
        },
    },
    "bio4": {
        "stir": {"required": {"beaker_idx", "count"}, "allowed": {"beaker_idx", "count"}},
        "tap": {"required": {"beaker_idx"}, "allowed": {"beaker_idx"}},
        "circle": {"required": {"beaker_idx"}, "allowed": {"beaker_idx"}},
        "swap": {
            "required": {"beaker_idx1", "beaker_idx2"},
            "allowed": {"beaker_idx1", "beaker_idx2"},
        },
    },
    "bio5": {
        "handoff": {
            "required": {"object", "from_arm", "to_arm"},
            "allowed": {"object", "from_arm", "to_arm"},
        },
        "place": {
            "required": {"object", "arm", "location"},
            "allowed": {"object", "arm", "location"},
        },
        "distractor_place": {
            "required": {"object", "location"},
            "allowed": {"object", "location"},
        },
        "pass_over": {
            "required": {"object", "arm"},
            "allowed": {"object", "arm"},
        },
        "fake_handoff": {"required": set(), "allowed": set()},
    },
    "ind2": {
        "lid_open": {
            "required": {"object", "close_after"},
            "allowed": {"object", "close_after"},
        },
        "lid_tap": {"required": {"object"}, "allowed": {"object"}},
    },
    "ind3": {
        "move_plug_to_rack": {
            "required": {"socket", "rack_socket"},
            "allowed": {"socket", "rack_socket"},
        },
        "move_plug_to_socket": {
            "required": {"from_socket", "to_socket"},
            "allowed": {"from_socket", "to_socket"},
        },
    },
    "ind5": {
        "handoff_to_box": {"required": {"object", "order"}, "allowed": {"object", "order"}},
        "direct_place_to_box": {
            "required": {"object", "order"},
            "allowed": {"object", "order"},
        },
    },
    "house1": {
        "cover": {"required": {"object"}, "allowed": {"object"}},
        "swap": {
            "required": {"count", "object1", "object2"},
            "allowed": {"count", "object1", "object2"},
        },
        "remove_occluder": {"required": set(), "allowed": set()},
    },
    "house3": {
        "place": {"required": {"order"}, "allowed": {"order"}},
        "swap": {
            "required": {"object1", "object2"},
            "allowed": {"object1", "object2"},
        },
    },
    "house5": {
        "handoff_and_place": {
            "required": {
                "layout_id",
                "object_index",
                "place_index",
                "grasp_index",
                "mode",
            },
            "allowed": {
                "layout_id",
                "object_index",
                "place_index",
                "grasp_index",
                "mode",
            },
        },
    },
}


def normalize_history_events(
    task_key: str,
    cfg: Mapping[str, Any],
    events: list[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return one project-wide public scene-event representation.

    Custom-task manifests already contain public event sequences. LPX task
    manifests hold their original reset parameters, so their public events are
    derived here without exposing evaluator-only targets.
    """
    public_events = [dict(event) for event in (events or []) if isinstance(event, Mapping)]
    if public_events:
        if task_key == "house5":
            return [_normalize_house5_event(event) for event in public_events]
        return public_events

    if task_key == "bio2":
        target = str(cfg.get("target_container", "C1"))
        result = [{"type": "pour", "source": "wash_bottle", "target": target}]
        distractor = cfg.get("distractor_event")
        if isinstance(distractor, Mapping):
            kind = str(distractor.get("type", "D1"))
            if kind == "D2":
                result.append(
                    {
                        "type": "swap_containers",
                        "object1": distractor.get("Cx"),
                        "object2": distractor.get("Cy"),
                    }
                )
            elif kind == "D3":
                result.append(
                    {
                        "type": "fake_pour",
                        "source": "wash_bottle",
                        "target": distractor.get("Cx"),
                    }
                )
        return result

    if task_key == "ind2":
        result = [
            {"type": "lid_open", "object": name, "close_after": True}
            for name in list(cfg.get("open_boxes", []) or [])
        ]
        distractor = cfg.get("distractor_event")
        if isinstance(distractor, Mapping) and distractor.get("type") == "tap_lid":
            result.append({"type": "lid_tap", "object": distractor.get("object")})
        return result

    if task_key == "ind3":
        result: list[dict[str, Any]] = []
        removed = cfg.get("remove_from")
        target = cfg.get("target_location_ids", cfg.get("target_socket", removed))
        if removed:
            result.append(
                {
                    "type": "move_plug_to_rack",
                    "socket": removed,
                    "rack_socket": target,
                }
            )
        distractor = cfg.get("distractor_event")
        loose = list(cfg.get("loose_peg", []) or [])
        if distractor:
            forbidden = {removed, target}
            if isinstance(distractor, str):
                forbidden.add(distractor)
            candidates = [
                name
                for name in loose
                if name not in forbidden
            ]
            if candidates:
                result.append(
                    {
                        "type": "move_plug_to_socket",
                        "from_socket": candidates[0],
                        "to_socket": candidates[0],
                    }
                )
        return result

    if task_key == "ind5":
        result = [
            {"type": "handoff_to_box", "object": name, "order": index}
            for index, name in enumerate(list(cfg.get("handoff", []) or []))
        ]
        distractor = cfg.get("distractor_event")
        if isinstance(distractor, Mapping) and distractor.get("object"):
            result.append(
                {
                    "type": "direct_place_to_box",
                    "object": distractor["object"],
                    "order": len(result),
                }
            )
        return result

    return public_events


def _normalize_house5_event(event: Mapping[str, Any]) -> dict[str, Any]:
    if event.get("type"):
        return dict(event)
    return {
        "type": "handoff_and_place",
        "layout_id": int(event.get("layout_id", 0)),
        "object_index": int(event.get("pass_id", 0)),
        "place_index": int(event.get("place_01", event.get("place_id", 0))),
        "grasp_index": int(event.get("grasp_id", event.get("pass_id", 0))),
        "mode": int(event.get("mode", 0)),
    }


def supported_event_types(task_key: str) -> set[str]:
    return set(EVENT_SCHEMAS.get(task_key, {}))


def validate_history_events(
    task_key: str,
    cfg: Mapping[str, Any],
    events: list[Mapping[str, Any]] | None,
    *,
    require_normalized: bool = False,
) -> list[str]:
    """Return validation errors for one public scene-only history.

    ``require_normalized`` is intended for the checked-in manifest. It rejects
    derived LPX reset parameters and unknown fields so all 10 task families
    expose one stable public event schema at runtime.
    """
    errors: list[str] = []
    if events is not None and not isinstance(events, list):
        return [f"{task_key} history events must be a list"]

    raw_events = list(events or [])
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            errors.append(f"{task_key} history event {index} must be a mapping")

    normalized = normalize_history_events(task_key, cfg, raw_events)
    if require_normalized and raw_events != normalized:
        errors.append(
            f"{task_key} history must use the canonical normalized event representation"
        )

    schemas = EVENT_SCHEMAS.get(task_key, {})
    for index, event in enumerate(normalized):
        event_type = event.get("type")
        schema = schemas.get(event_type)
        if schema is None:
            errors.append(
                f"{task_key} history event {index} has unsupported type {event_type!r}"
            )
            continue

        fields = set(event) - {"type"}
        missing = sorted(schema["required"] - fields)
        if missing:
            errors.append(
                f"{task_key} history event {index} {event_type!r} is missing {missing}"
            )
        unknown = sorted(fields - schema["allowed"])
        if unknown:
            errors.append(
                f"{task_key} history event {index} {event_type!r} has unsupported fields {unknown}"
            )
        errors.extend(_validate_event_values(task_key, index, event))
    return errors


def _validate_event_values(
    task_key: str, index: int, event: Mapping[str, Any]
) -> list[str]:
    """Validate the public identifiers used by a canonical scene event."""
    event_type = str(event.get("type"))
    prefix = f"{task_key} history event {index} {event_type!r}"
    errors: list[str] = []

    def label(field: str, expected_prefix: str) -> None:
        value = event.get(field)
        if not _is_label(value, expected_prefix):
            errors.append(f"{prefix} field {field!r} must be a {expected_prefix}-label")

    def integer(field: str, *, minimum: int = 0, maximum: int | None = None) -> None:
        value = event.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{prefix} field {field!r} must be an integer")
            return
        if value < minimum or (maximum is not None and value > maximum):
            bound = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
            errors.append(f"{prefix} field {field!r} must be in {bound}")

    if task_key == "bio2":
        if event_type in {"pour", "fake_pour"}:
            if event.get("source") != "wash_bottle":
                errors.append(f"{prefix} field 'source' must be 'wash_bottle'")
            label("target", "C")
        elif event_type == "swap_containers":
            label("object1", "C")
            label("object2", "C")
    elif task_key == "bio4":
        for field in ("beaker_idx", "beaker_idx1", "beaker_idx2"):
            if field in event:
                integer(field, minimum=1, maximum=4)
        if "count" in event:
            integer("count", minimum=1)
    elif task_key == "bio5":
        if "object" in event:
            label("object", "T")
        for field in ("arm", "from_arm", "to_arm"):
            if field in event and event[field] not in {"left", "right"}:
                errors.append(f"{prefix} field {field!r} must be 'left' or 'right'")
        if "location" in event and not _is_tube_location(event["location"]):
            errors.append(f"{prefix} field 'location' must be D1..D3 or DD1..DD3")
    elif task_key == "ind2":
        label("object", "B")
        if event_type == "lid_open" and not isinstance(event.get("close_after"), bool):
            errors.append(f"{prefix} field 'close_after' must be a boolean")
    elif task_key == "ind3":
        for field in ("socket", "from_socket", "to_socket", "rack_socket"):
            if field in event:
                label(field, "S")
    elif task_key == "ind5":
        label("object", "P")
        integer("order", minimum=0)
    elif task_key == "house1":
        for field in ("object", "object1", "object2"):
            if field in event:
                label(field, "O")
        if "count" in event:
            integer("count", minimum=1)
    elif task_key == "house3":
        if event_type == "place":
            order = event.get("order")
            if not isinstance(order, list) or not order:
                errors.append(f"{prefix} field 'order' must be a non-empty list")
            elif len(order) != len(set(order)) or any(
                not _is_label(value, "O") for value in order
            ):
                errors.append(f"{prefix} field 'order' must contain unique O-labels")
        else:
            label("object1", "O")
            label("object2", "O")
    elif task_key == "house5":
        for field in ("layout_id", "object_index", "place_index", "grasp_index"):
            integer(field, minimum=0)
        integer("mode", minimum=0, maximum=1)
    return errors


def _is_label(value: Any, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(prefix):
        return False
    suffix = value[len(prefix) :]
    return bool(suffix) and suffix.isdecimal() and int(suffix) >= 1


def _is_tube_location(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    prefix = "DD" if value.startswith("DD") else "D"
    return _is_label(value, prefix) and int(value[len(prefix) :]) in {1, 2, 3}


class DemonstrationReplayError(RuntimeError):
    """A reset-time failure while producing the required scene demonstration."""


class _Bio5ReplayRecorder:
    """Collector-compatible label sink used by source bio5 controls.

    The benchmark intentionally retains no collector artifacts or labels; the
    source functions only need this method while executing their trajectory.
    """

    def set_subtask(self, _: str) -> None:
        return


class SceneDemonstrationReplayer:
    """Replays public scene events and captures the policy camera after each frame."""

    def __init__(
        self,
        *,
        task_key: str,
        raw_env: Any,
        cfg: Mapping[str, Any],
        events: list[Mapping[str, Any]] | None,
        resolve_actor: Callable[[Any], Any | None],
        on_frame: Callable[[bool], None] | None = None,
        source_camera: str | None = None,
        initial_frame_step: Callable[[], None] | None = None,
    ) -> None:
        self.task_key = task_key
        self.raw_env = raw_env
        self.cfg = dict(cfg)
        self.events = normalize_history_events(task_key, cfg, events)
        self.resolve_actor = resolve_actor
        self.on_frame = on_frame
        self.source_camera = source_camera
        self.initial_frame_step = initial_frame_step
        self.demo_frame_stride = int(
            DEMO_FRAME_STRIDES.get(task_key, DEFAULT_DEMO_FRAME_STRIDE)
        )
        if self.demo_frame_stride <= 0:
            raise ValueError("demo frame stride must be positive")
        self._raw_demo_frame_count = 0
        self.frames: list[np.ndarray] = []
        self.camera: str | None = None
        self._ind5_planners: tuple[Any, Any] | None = None
        self._bio5_planners: tuple[Any, Any] | None = None
        self._bio5_handoff: dict[str, Any] | None = None
        self._bio5_replay_warnings: list[str] = []
        self._house5_planners: tuple[Any, Any] | None = None
        self._house5_replay_warnings: list[str] = []
        self._house5_replay_mode = "planner"

    def replay(self) -> dict[str, Any]:
        if self.task_key in {"bio4", "house1", "house3"}:
            # The custom collectors stabilize the scene first, then record ten
            # stationary source-camera frames before their scene event sequence.
            for _ in range(10):
                if self.initial_frame_step is not None:
                    self.initial_frame_step()
                self._append_demo_frame()
        if self.task_key == "ind2":
            self._initialize_ind2_lids()
        if self.task_key == "ind5":
            self._initialize_ind5_motion_planners()
        if self.task_key == "bio5":
            self._replay_bio5_source_trajectory()
        else:
            if self.task_key == "house5":
                self._initialize_house5_motion_planners()
            for event in self.events:
                self._replay_event(event)
        if self.task_key == "bio4":
            # The collector calls return_rod_to_initial once after every task
            # sequence, including sequences whose final event already did so.
            self._return_bio4_rod_to_initial()
        interaction_frame = self._capture_frame(is_demonstration=False)
        self.frames.append(interaction_frame)
        return {
            "frames": np.stack(self.frames, axis=0),
            "demo_frame_count": len(self.frames) - 1,
            "raw_demo_frame_count": self._raw_demo_frame_count,
            "demo_frame_stride": self.demo_frame_stride,
            "interaction_frame_index": len(self.frames) - 1,
            "camera": self.camera,
            "events": deepcopy(self.events),
            "timing_source": HISTORY_TIMING_SOURCE,
            "replay_warnings": list(self._house5_replay_warnings)
            + list(self._bio5_replay_warnings),
            "replay_mode": "planner" if self.task_key == "bio5" else self._house5_replay_mode,
        }

    def _replay_bio5_source_trajectory(self) -> None:
        """Run the vendored L5 collector's control sequence without its I/O path.

        Bio5 replays only the source collector's public demonstration prefix.
        The target-tube transfer after the listed events is the policy
        interaction and must remain untouched at the handoff boundary.
        """
        if not self._bio5_has_bimanual_agents():
            raise DemonstrationReplayError(
                "bio5 source replay requires exactly two Piper-X agents; "
                "scene-pose fallback is intentionally disabled"
            )

        from mani_skill.envs.tasks.custom_task.biolab import run_scene_L5_02 as source
        from scipy.spatial.transform import Rotation as R

        record_id = self.cfg.get("source_record_id")
        if not isinstance(record_id, int):
            raise DemonstrationReplayError("bio5 source replay requires source_record_id")
        if not 0 <= record_id < len(source.TASK_SEQUENCES):
            raise DemonstrationReplayError(f"invalid bio5 source_record_id {record_id}")

        task = source.TASK_SEQUENCES[record_id]
        expected_events = task["sequence"]
        if list(self.events) != expected_events:
            raise DemonstrationReplayError(
                f"bio5 manifest events do not match source task {record_id}"
            )
        if int(self.cfg.get("layout_id", -1)) != int(task["layout"]):
            raise DemonstrationReplayError(
                f"bio5 manifest layout does not match source task {record_id}"
            )
        if int(self.cfg.get("repeat_id", -1)) != int(task["repeat_id"]):
            raise DemonstrationReplayError(
                f"bio5 manifest repeat_id does not match source task {record_id}"
            )

        # The source collector starts with 50 unrecorded held-arm steps and
        # records ten more before dispatching the task sequence.
        source.improve_grip_physics(self.raw_env)
        for _ in range(50):
            self.raw_env.step(self._bio5_hold_action())
        for _ in range(10):
            self._bio5_record_step(self._bio5_hold_action())

        # The collector caches planning solvers globally. Reset that cache so
        # a reset cannot reuse planners tied to an earlier environment.
        source._g_solvers[:] = [None, None]
        recorder = _Bio5ReplayRecorder()
        agent = self.raw_env.agent
        handoff_state: dict[str, Any] = {}
        right_grasp_quat_global: np.ndarray | None = None
        record_step = self._bio5_record_step

        for step in expected_events:
            step_type = step["type"]
            if step_type == "handoff":
                tube_index = int(step["object"][1]) - 1
                z_axis = np.array([0, -1, 0])
                x_axis = np.array([0, 0, -1])
                handover_quat = R.from_matrix(
                    np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
                ).as_quat()
                success, left_index, ee_pos, ee_quat, qpos = source.pass_to(
                    env=self.raw_env,
                    tube_idx=tube_index,
                    robot_idx=0,
                    target_pos=np.array([-0.4, 0.0, 0.25], dtype=np.float32),
                    target_quat=handover_quat,
                    record_step=record_step,
                    recorder=recorder,
                    grasp_rot_deg=np.array([0.0, 45.0, 0.0], dtype=np.float32),
                )
                if not success:
                    raise DemonstrationReplayError("bio5 source pass_to failed")

                tube_pos, tube_quat = source.get_tube_pose(self.raw_env, tube_index)
                if tube_pos is not None and tube_quat is not None:
                    tube_rotation = R.from_quat(
                        [tube_quat[1], tube_quat[2], tube_quat[3], tube_quat[0]]
                    )
                    correction, _ = R.align_vectors([[0, 0, 1]], [tube_rotation.apply([0, 0, 1])])
                    tcp_quat = _to_vector(agent.agents[0].tcp_pose.q, 4)
                    adjusted = correction * R.from_quat(
                        [tcp_quat[1], tcp_quat[2], tcp_quat[3], tcp_quat[0]]
                    )
                    adjusted_scipy = adjusted.as_quat()
                    handover_quat = np.array(
                        [adjusted_scipy[3], *adjusted_scipy[:3]], dtype=np.float32
                    )
                    source.move_to_position_only(
                        self.raw_env,
                        robot_idx=0,
                        target_pos=_to_vector(agent.agents[0].tcp_pose.p, 3),
                        target_quat=handover_quat,
                        steps=60,
                        record_step=record_step,
                        gripper_open=False,
                        allow_adaptive_search=True,
                        limit_search_range=True,
                    )

                success, right_pos, right_quat, right_qpos, handover_quat = source.right_pass(
                    env=self.raw_env,
                    tube_idx=tube_index,
                    robot_idx=left_index,
                    initial_ee_pos=ee_pos,
                    initial_ee_quat=ee_quat,
                    initial_qpos=qpos,
                    recorder=recorder,
                    record_step=record_step,
                )
                if not success:
                    raise DemonstrationReplayError("bio5 source right_pass failed")
                right_index = 1 - left_index
                handoff_state = {
                    "left_index": left_index,
                    "right_index": right_index,
                    "tube_index": tube_index,
                    "right_pos": right_pos,
                    "right_quat": right_quat,
                    "right_qpos": right_qpos,
                    "handover_quat": handover_quat,
                }
            elif step_type == "place":
                if not handoff_state:
                    continue
                source.right_place_procedure(
                    env=self.raw_env,
                    tube_idx=handoff_state["tube_index"],
                    right_arm_idx=handoff_state["right_index"],
                    left_arm_idx=handoff_state["left_index"],
                    right_robot=agent.agents[handoff_state["right_index"]],
                    left_robot=agent.agents[handoff_state["left_index"]],
                    right_handover_quat=handoff_state["handover_quat"],
                    right_initial_pos=handoff_state["right_pos"],
                    right_initial_quat=handoff_state["right_quat"],
                    right_initial_qpos=handoff_state["right_qpos"],
                    right_place_idx=int(step["location"][1]) - 1,
                    recorder=recorder,
                    record_step=record_step,
                )
                handoff_state = {}
            elif step_type == "distractor_place":
                tube_index = int(step["object"][1]) - 1
                right_grasp_quat_global = R.from_euler("xyz", np.deg2rad([0, 60, 0])).as_quat().astype(np.float32)
                source.grasp_and_place(
                    env=self.raw_env,
                    tube_idx=tube_index,
                    robot_idx=1,
                    other_robot_idx=0,
                    robot=agent.agents[1],
                    other_robot=agent.agents[0],
                    start_idx=tube_index,
                    end_idx=5 + int(step["location"][2]) - 1,
                    right_handover_quat=right_grasp_quat_global,
                    recorder=recorder,
                    record_step=record_step,
                )
            elif step_type == "fake_handoff":
                source.fake_pass(
                    env=self.raw_env, robot_idx=0, record_step=record_step, recorder=recorder
                )
            elif step_type == "pass_over":
                source.fake_place(
                    env=self.raw_env,
                    tube_idx=int(step["object"][1]) - 1,
                    robot=agent.agents[1],
                    robot_idx=1,
                    other_robot_idx=0,
                    other_robot=agent.agents[0],
                    record_step=record_step,
                )


    def _bio5_hold_action(self) -> dict[str, np.ndarray]:
        action = {
            uid: np.zeros(space.shape, dtype=np.float32)
            for uid, space in self.raw_env.action_space.items()
        }
        for index, robot in enumerate(self.raw_env.agent.agents):
            uid = f"piper_x-{index}"
            if uid not in action:
                continue
            qpos = _to_vector(robot.robot.get_qpos(), None)
            action[uid] = np.concatenate([qpos[:6], np.asarray([-1.0], dtype=np.float32)])
        return action

    def _bio5_record_step(self, action: Any, mock_obs: Any | None = None) -> Any:
        if mock_obs is None:
            result = self.raw_env.step(action)
        else:
            result = (mock_obs, 0.0, False, False, {})
        self._capture_source_step()
        return result

    def _replay_event(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        if self.task_key == "house5" and event_type == "handoff_and_place":
            self._replay_house5_handoff(event)
            return
        if self.task_key == "house3" and event_type == "place":
            self._house3_place(event)
            return
        handlers = {
            "pour": self._pour,
            "swap_containers": self._swap_containers,
            "fake_pour": self._fake_pour,
            "lid_open": self._lid_open,
            "lid_tap": self._lid_tap,
            "move_plug_to_rack": self._move_plug_to_rack,
            "move_plug_to_socket": self._move_plug_to_socket,
            "handoff_to_box": self._handoff_to_box,
            "direct_place_to_box": self._direct_place_to_box,
            "stir": self._stir,
            "tap": self._tap,
            "circle": self._circle,
            "swap": self._swap,
            "handoff": self._tube_handoff,
            "place": self._tube_place,
            "distractor_place": self._tube_place,
            "pass_over": self._tube_pass_over,
            "fake_handoff": self._tube_fake_handoff,
            "cover": self._cover,
            "remove_occluder": self._remove_occluder,
        }
        handler = handlers.get(event_type)
        if handler is None:
            raise ValueError(f"Unsupported {self.task_key} demonstration event: {event_type!r}")
        handler(event)

    def _capture_frame(
        self, *, is_demonstration: bool = True, notify_callback: bool = True
    ) -> np.ndarray:
        try:
            obs = self.raw_env.get_obs()
            camera, rgb = primary_rgb_from_obs(obs, preferred_camera=self.source_camera)
        except Exception:
            self.raw_env.scene.update_render(
                update_sensors=True, update_human_render_cameras=False
            )
            self.raw_env.capture_sensor_data()
            sensor_data = {
                name: sensor.get_obs()
                for name, sensor in self.raw_env.scene.sensors.items()
            }
            obs = {"sensor_data": sensor_data}
            camera, rgb = primary_rgb_from_obs(obs, preferred_camera=self.source_camera)
        if self.camera is None:
            self.camera = camera
        elif self.camera != camera:
            raise RuntimeError(
                f"Policy camera changed during demonstration: {self.camera!r} -> {camera!r}"
            )
        if notify_callback and self.on_frame is not None:
            self.on_frame(is_demonstration)
        return rgb

    def _capture_source_step(self) -> None:
        """Capture one real LPX planner control step without exposing its info."""
        self._append_demo_frame()

    def _append_demo_frame(self) -> None:
        """Capture one source frame and retain it according to the v4 stride.

        The simulator still executes every source control step.  Only the
        policy-facing/video frame sequence is thinned, so object state,
        evaluator baselines, and task timing remain unchanged.
        """
        # Do not render or notify the video callback for discarded source
        # frames. v4 defines acceleration as temporal downsampling of the demo
        # stream, so the callback (and therefore recorded videos) sees the
        # same retained sequence as the policy. The simulator state has
        # already been advanced by the caller and remains untouched here.
        retain = self._raw_demo_frame_count % self.demo_frame_stride == 0
        if retain:
            frame = self._capture_frame(notify_callback=False)
            self.frames.append(frame)
            if self.on_frame is not None:
                self.on_frame(True)
        self._raw_demo_frame_count += 1

    def _actor(self, object_ref: Any) -> Any:
        actor = self.resolve_actor(object_ref)
        if actor is None:
            raise KeyError(f"Could not resolve demonstration object {object_ref!r}")
        return actor

    def _actor_pose(self, actor: Any) -> tuple[np.ndarray, np.ndarray]:
        pose = actor.pose
        return _to_vector(pose.p, 3), _to_vector(pose.q, 4)

    def _set_actor_pose(
        self, actor: Any, position: np.ndarray, quaternion: np.ndarray
    ) -> None:
        device = getattr(self.raw_env, "device", None)
        if device is not None and hasattr(actor, "set_pose"):
            try:
                from mani_skill.utils.structs.pose import Pose

                p = torch.as_tensor(position, dtype=torch.float32, device=device).reshape(1, 3)
                q = torch.as_tensor(quaternion, dtype=torch.float32, device=device).reshape(1, 4)
                actor.set_pose(Pose.create_from_pq(p, q))
                return
            except Exception:
                pass
        actor.set_pose(sapien.Pose(p=position, q=quaternion))

    def _move_actor(
        self,
        actor: Any,
        targets: list[np.ndarray],
        *,
        steps_per_target: int,
        arc_height: float = 0.0,
    ) -> None:
        current, quat = self._actor_pose(actor)
        for target in targets:
            target = np.asarray(target, dtype=np.float32)
            start = current.copy()
            for frame_index in range(steps_per_target):
                alpha = (frame_index + 1) / steps_per_target
                position = (1.0 - alpha) * start + alpha * target
                if arc_height:
                    position[2] += arc_height * np.sin(np.pi * alpha)
                self._set_actor_pose(actor, position, quat)
                self._append_demo_frame()
            current = target

    def _move_actor_inclusive(
        self,
        actor: Any,
        target: np.ndarray,
        *,
        interpolation_steps: int,
        arc_height: float = 0.0,
    ) -> None:
        """Replay custom-task loops that capture both segment endpoints."""
        if interpolation_steps <= 0:
            raise ValueError("interpolation_steps must be positive")
        start, quat = self._actor_pose(actor)
        target = np.asarray(target, dtype=np.float32)
        for frame_index in range(interpolation_steps + 1):
            alpha = frame_index / interpolation_steps
            position = (1.0 - alpha) * start + alpha * target
            if arc_height:
                position[2] += arc_height * np.sin(np.pi * alpha)
            self._set_actor_pose(actor, position, quat)
            self._append_demo_frame()

    def _move_actor_source_trajectory(
        self,
        actor: Any,
        points: list[np.ndarray],
        *,
        steps_per_point: int,
    ) -> None:
        """Replay LPX helper timing: every listed point owns one interpolation segment."""
        for point in points:
            self._move_actor(actor, [np.asarray(point, dtype=np.float32)], steps_per_target=steps_per_point)

    def _swap_actors(
        self,
        actor1: Any,
        actor2: Any,
        *,
        steps: int = 20,
        asymmetric: bool = False,
        lift_height: float = 0.15,
    ) -> None:
        pos1, quat1 = self._actor_pose(actor1)
        pos2, quat2 = self._actor_pose(actor2)
        for frame_index in range(steps + 1):
            alpha = frame_index / steps
            new1 = (1.0 - alpha) * pos1 + alpha * pos2
            new2 = (1.0 - alpha) * pos2 + alpha * pos1
            new1[2] += lift_height * np.sin(np.pi * alpha)
            if asymmetric:
                new2[2] = max(float(new2[2] - 0.03 * np.sin(np.pi * alpha)), 0.02)
            else:
                new2[2] += lift_height * np.sin(np.pi * alpha)
            self._set_actor_pose(actor1, new1, quat1)
            self._set_actor_pose(actor2, new2, quat2)
            self._append_demo_frame()

    def _pour(self, event: Mapping[str, Any]) -> None:
        source = self._actor(event.get("source", "wash_bottle"))
        target = self._actor(event.get("target"))
        source_pos, _ = self._actor_pose(source)
        target_pos, _ = self._actor_pose(target)
        self._move_actor_source_trajectory(
            source,
            [
                source_pos,
                [source_pos[0], source_pos[1], 0.15],
                [target_pos[0] - 0.17, target_pos[1], 0.05],
                [target_pos[0] - 0.17, target_pos[1], 0.03],
                [target_pos[0] - 0.17, target_pos[1], 0.05],
                [source_pos[0], source_pos[1], 0.15],
                source_pos,
            ],
            steps_per_point=30,
        )

    def _swap_containers(self, event: Mapping[str, Any]) -> None:
        name1, name2 = event.get("object1"), event.get("object2")
        if not name1 or not name2:
            return
        actor1 = self._actor(name1)
        actor2 = self._actor(name2)
        pos1, _ = self._actor_pose(actor1)
        pos2, _ = self._actor_pose(actor2)
        # Keep the beaker above the table while it waits between swap legs.
        temporary = np.asarray([0.2, 0.0, 0.10], dtype=np.float32)
        self._move_actor_source_trajectory(
            actor1,
            [pos1, [pos1[0], pos1[1], pos1[2] + 0.1], temporary],
            steps_per_point=25,
        )
        self._move_actor_source_trajectory(
            actor2,
            [pos2, [pos2[0], pos2[1], pos2[2] + 0.1], [pos1[0], pos1[1], pos1[2] + 0.1], pos1],
            steps_per_point=25,
        )
        self._move_actor_source_trajectory(
            actor1,
            [temporary, [pos2[0], pos2[1], pos2[2] + 0.1], pos2],
            steps_per_point=25,
        )
        self._step_scene_and_capture(15)

    def _fake_pour(self, event: Mapping[str, Any]) -> None:
        target_name = event.get("target")
        if not target_name:
            return
        source = self._actor(event.get("source", "wash_bottle"))
        target = self._actor(target_name)
        source_pos, _ = self._actor_pose(source)
        target_pos, _ = self._actor_pose(target)
        hover = np.asarray([target_pos[0] - 0.17, target_pos[1], 0.12], dtype=np.float32)
        self._move_actor_source_trajectory(
            source,
            [source_pos, [source_pos[0], source_pos[1], 0.15], hover],
            steps_per_point=20,
        )
        self._move_actor_source_trajectory(source, [hover, hover], steps_per_point=50)
        self._move_actor_source_trajectory(
            source,
            [hover, [source_pos[0], source_pos[1], 0.15], source_pos],
            steps_per_point=20,
        )

    def _lid_open(self, event: Mapping[str, Any]) -> None:
        box = self._actor(event.get("object"))
        lid_index = self._lid_joint_index(box)
        if lid_index is None:
            raise RuntimeError(f"Could not find lid joint for {event.get('object')!r}")
        self._animate_ind2_lid(box, lid_index, target_angle=1.57)
        if event.get("close_after", False):
            self._animate_ind2_lid(box, lid_index, target_angle=0.0)

    def _lid_tap(self, event: Mapping[str, Any]) -> None:
        # This design-sheet marker has no corresponding collector motion.
        self._actor(event.get("object"))

    def _initialize_ind2_lids(self) -> None:
        """Replay the collector's per-box close loop before inspection starts."""
        for box in self.raw_env.box[: self.raw_env.num_boxes]:
            lid_index = self._lid_joint_index(box)
            if lid_index is None:
                raise RuntimeError(f"Could not find lid joint for {box.name!r}")
            self._animate_ind2_lid(box, lid_index, target_angle=0.0)

    def _move_plug_to_rack(self, event: Mapping[str, Any]) -> None:
        socket = str(event.get("socket"))
        plug = self.raw_env.pegs[_label_index(socket, "S")]
        rack_socket = str(event.get("rack_socket"))
        rack = self.raw_env.plug_racks[_label_index(rack_socket, "S")]
        plug_pos, _ = self._actor_pose(plug)
        rack_pos, _ = self._actor_pose(rack)
        target = np.asarray([rack_pos[0], rack_pos[1], self.raw_env._rack_outer_hs[2] * 2])
        self._set_actor_pose(plug, plug_pos, PLUG_SOURCE_QUATERNION)
        self._hold(20)
        self._move_actor(
            plug,
            [
                [plug_pos[0], plug_pos[1], plug_pos[2] + 0.15],
                [target[0], target[1], plug_pos[2] + 0.15],
                target,
            ],
            steps_per_target=20,
        )

    def _move_plug_to_socket(self, event: Mapping[str, Any]) -> None:
        from_socket = str(event.get("from_socket"))
        to_socket = str(event.get("to_socket"))
        plug = self.raw_env.pegs[_label_index(from_socket, "S")]
        socket = self.raw_env.boxes[_label_index(to_socket, "S")]
        plug_pos, _ = self._actor_pose(plug)
        socket_pos, _ = self._actor_pose(socket)
        target = np.asarray([socket_pos[0], socket_pos[1], 0.03], dtype=np.float32)
        self._set_actor_pose(plug, plug_pos, PLUG_SOURCE_QUATERNION)
        self._hold(20)
        self._move_actor(
            plug,
            [
                [plug_pos[0], plug_pos[1], plug_pos[2] + 0.15],
                [target[0], target[1], plug_pos[2] + 0.15],
                target,
            ],
            steps_per_target=20,
        )

    def _handoff_to_box(self, event: Mapping[str, Any]) -> None:
        self._replay_ind5_pack(event, direct=False)

    def _direct_place_to_box(self, event: Mapping[str, Any]) -> None:
        self._replay_ind5_pack(event, direct=True)

    def _replay_ind5_pack(self, event: Mapping[str, Any], *, direct: bool) -> None:
        """Execute one source ``ind5`` packing state machine with two Piper-X arms."""
        env = self.raw_env
        if not (hasattr(env, "left_arm") and hasattr(env, "right_arm")):
            raise DemonstrationReplayError("ind5 requires a bimanual Piper-X environment")

        actor = self._actor(event.get("object"))
        order = int(event.get("order", 0))
        target = IND5_PACKING_PLACE_POSITIONS[order % len(IND5_PACKING_PLACE_POSITIONS)]

        if self._ind5_planners is None:
            raise DemonstrationReplayError("ind5 planners were not initialized")
        left, right = self._ind5_planners

        left.set_passive_gripper_state(left.OPEN)
        left.set_passive_arm_target_qpos(self._arm_qpos("right"))

        cube_pos, _ = self._actor_pose(actor)
        grasp_pose = env.left_arm.build_grasp_pose(
            IND5_SOURCE_APPROACHING, IND5_SOURCE_CLOSING_LEFT, cube_pos
        ) * sapien.Pose(q=[0, 0, 0, 1])
        vertical_grasp = env.left_arm.build_grasp_pose(
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            IND5_SOURCE_CLOSING_LEFT,
            cube_pos,
        ) * sapien.Pose(q=[0, 0, 0, 1])
        vertical_grasp = sapien.Pose(
            p=[cube_pos[0], cube_pos[1], cube_pos[2] - 0.015], q=vertical_grasp.q
        )

        left.open_gripper()
        self._ind5_move(left, sapien.Pose(p=[-0.3, 0.3, 0.15], q=grasp_pose.q))
        self._ind5_move(left, sapien.Pose(p=[cube_pos[0], cube_pos[1], 0.12], q=grasp_pose.q))
        self._ind5_move(left, vertical_grasp)
        left.close_gripper()
        self._ind5_move(
            left, sapien.Pose(p=[cube_pos[0], cube_pos[1], cube_pos[2] + 0.12], q=grasp_pose.q)
        )

        if direct:
            above_target = sapien.Pose(p=[target[0], target[1], target[2] + 0.10], q=grasp_pose.q)
            self._ind5_move(left, above_target)
            self._ind5_move(left, sapien.Pose(p=target, q=grasp_pose.q))
            left.open_gripper()
            self._ind5_move(left, above_target)
            self._ind5_move(left, sapien.Pose(p=[-0.3, 0.3, 0.2], q=grasp_pose.q))
            left.clear_passive_gripper_state()
            left.clear_passive_arm_target_qpos()
            return

        handoff_pose_left = env.left_arm.build_grasp_pose(
            IND5_HANDOFF_APPROACHING_LEFT, IND5_HANDOFF_CLOSING, IND5_HANDOFF_POSITION
        )
        repeat_id = int(self.cfg.get("repeat_id", 1))
        if repeat_id == 2:
            handoff_pose_left = handoff_pose_left * sapien.Pose(q=[np.cos(np.pi / 12), 0, 0, np.sin(np.pi / 12)])
        elif repeat_id == 3:
            handoff_pose_left = handoff_pose_left * sapien.Pose(q=[np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)])
        self._ind5_move(left, handoff_pose_left)
        left.clear_passive_gripper_state()
        left.clear_passive_arm_target_qpos()

        right.set_passive_arm_target_qpos(self._arm_qpos("left"))
        right.open_gripper()
        handoff_pose_right = env.right_arm.build_grasp_pose(
            IND5_HANDOFF_APPROACHING_RIGHT, IND5_HANDOFF_CLOSING, IND5_HANDOFF_POSITION
        )
        handoff_position_right = IND5_HANDOFF_POSITION.copy()
        handoff_position_right[2] += 0.01
        handoff_pose_right = handoff_pose_right * sapien.Pose(q=[0.7071, 0, 0, -0.7071])
        if repeat_id == 2:
            handoff_pose_right = handoff_pose_right * sapien.Pose(q=[np.cos(-np.pi / 12), 0, 0, np.sin(-np.pi / 12)])
        elif repeat_id == 3:
            handoff_pose_right = handoff_pose_right * sapien.Pose(q=[np.cos(-np.pi / 8), 0, 0, np.sin(-np.pi / 8)])
        pre_handoff = sapien.Pose(
            p=[handoff_position_right[0], handoff_position_right[1] - 0.05, handoff_position_right[2]],
            q=handoff_pose_right.q,
        )
        self._ind5_move(right, pre_handoff)
        self._ind5_move(right, handoff_pose_right)
        right.close_gripper()
        right.clear_passive_arm_target_qpos()

        left.set_passive_arm_target_qpos(self._arm_qpos("right"))
        left.open_gripper()
        left.clear_passive_arm_target_qpos()

        right.set_passive_gripper_state(right.OPEN)
        self._ind5_move(right, pre_handoff)
        left.set_passive_arm_target_qpos(self._arm_qpos("right"))
        if repeat_id == 1:
            self._ind5_move(left, sapien.Pose(p=[-0.45, 0, 0.2], q=handoff_pose_left.q))
        self._ind5_move(left, sapien.Pose(p=[-0.3, 0.3, 0.2], q=grasp_pose.q))
        left.clear_passive_arm_target_qpos()

        above_target = sapien.Pose(p=[target[0], target[1], target[2] + 0.10], q=grasp_pose.q)
        self._ind5_move(right, above_target)
        self._ind5_move(right, sapien.Pose(p=target, q=grasp_pose.q))
        right.open_gripper()
        self._ind5_move(right, above_target)
        self._ind5_move(right, sapien.Pose(p=[-0.4, -0.3, 0.15], q=grasp_pose.q))
        right.clear_passive_gripper_state()

    def _initialize_ind5_motion_planners(self) -> None:
        from mani_skill.examples.motionplanning.double_piper_x.motionplanner import (
            PiperXArmMotionPlanningSolver,
        )

        env = self.raw_env
        if not (hasattr(env, "left_arm") and hasattr(env, "right_arm")):
            raise DemonstrationReplayError("ind5 requires a bimanual Piper-X environment")
        for _ in range(40):
            self._ind5_hold_arms_closed()
        left = PiperXArmMotionPlanningSolver(
            env,
            debug=False,
            vis=False,
            base_pose=env.left_arm.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            arm="left",
        )
        right = PiperXArmMotionPlanningSolver(
            env,
            debug=False,
            vis=False,
            base_pose=env.right_arm.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            arm="right",
        )
        self._configure_ind5_capture(left)
        self._configure_ind5_capture(right)
        right_qpos = self._arm_qpos("right")
        left_qpos = self._arm_qpos("left")
        left.set_passive_arm_target_qpos(right_qpos)
        left.close_gripper()
        left.clear_passive_arm_target_qpos()
        right.set_passive_arm_target_qpos(left_qpos)
        right.close_gripper()
        right.clear_passive_arm_target_qpos()
        for _ in range(50):
            self._ind5_hold_arms_closed()
        self._ind5_planners = (left, right)

    def _configure_ind5_capture(self, planner: Any) -> None:
        original_step = planner._step_env

        def capture_step(action: Any):
            result = original_step(action)
            self._capture_source_step()
            return result

        planner._step_env = capture_step

    def _ind5_hold_arms_closed(self) -> None:
        action = {
            "piper_x-0": np.hstack([self._arm_qpos("left")[:6], -1.0]),
            "piper_x-1": np.hstack([self._arm_qpos("right")[:6], -1.0]),
        }
        self.raw_env.step(action)

    def _arm_qpos(self, arm: str) -> np.ndarray:
        robot = self.raw_env.left_arm.robot if arm == "left" else self.raw_env.right_arm.robot
        return _to_vector(robot.get_qpos(), None).copy()

    @staticmethod
    def _ind5_move(planner: Any, pose: sapien.Pose) -> Any:
        result = planner.move_to_pose_with_screw(pose)
        if result == -1:
            result = planner.move_to_pose_with_RRTConnect(pose)
        if result == -1:
            raise DemonstrationReplayError("ind5 source motion plan failed")
        return result

    def _stir(self, event: Mapping[str, Any]) -> None:
        beaker = self._actor(f"beaker{int(event.get('beaker_idx', 1))}")
        rod = self._actor("glass_rod1")
        center, _ = self._actor_pose(beaker)
        count = max(1, int(event.get("count", 1)))
        self._circle_actor(
            rod,
            center=center + np.asarray([0.0, 0.0, 0.08]),
            radius=0.03,
            circles=count,
            lift_after_each=True,
        )

    def _tap(self, event: Mapping[str, Any]) -> None:
        beaker = self._actor(f"beaker{int(event.get('beaker_idx', 1))}")
        rod = self._actor("glass_rod1")
        center, _ = self._actor_pose(beaker)
        above = center + np.asarray([0.05, 0.0, 0.15], dtype=np.float32)
        down = center + np.asarray([0.05, 0.0, 0.05], dtype=np.float32)
        # Source rod_drop_to_beaker first assigns [0, 0, 0] Euler rotation.
        # This assignment is not captured; its first recorded frame is the
        # start of rod_drop_between_positions.
        self._set_actor_pose(rod, above, BIO4_DROP_QUATERNION)
        # ``rod_drop_between_positions`` records range(15 + 1), then the
        # source returns home with range(10 + 1). Setting ``above`` itself is
        # not a recorded operation; it becomes the first descent frame.
        self._move_actor_inclusive(rod, down, interpolation_steps=15)
        self._move_actor_inclusive_with_quaternion(
            rod,
            BIO4_ROD_HOME_POSITION,
            BIO4_HOME_QUATERNION,
            interpolation_steps=10,
        )

    def _circle(self, event: Mapping[str, Any]) -> None:
        beaker = self._actor(f"beaker{int(event.get('beaker_idx', 1))}")
        rod = self._actor("glass_rod1")
        center, _ = self._actor_pose(beaker)
        # draw_circle_above_beaker calls set_rod_pose([0, 90, 0]) before the
        # recorded stir operation. Its scipy quaternion order is intentional.
        rod_position, _ = self._actor_pose(rod)
        self._set_actor_pose(rod, rod_position, BIO4_VERTICAL_ROD_QUATERNION)
        self._circle_actor(
            rod,
            center=center + np.asarray([0.0, 0.0, 0.15]),
            radius=0.05,
            circles=1,
            lift_after_each=True,
        )
        self._return_bio4_rod_to_initial()

    def _swap(self, event: Mapping[str, Any]) -> None:
        if self.task_key == "bio4":
            actor1 = self._actor(f"beaker{int(event.get('beaker_idx1', 1))}")
            actor2 = self._actor(f"beaker{int(event.get('beaker_idx2', 2))}")
        elif self.task_key == "house1":
            actor1 = self._house_actor(event.get("object1"))
            actor2 = self._house_actor(event.get("object2"))
        elif self.task_key == "house3":
            actor1 = self._house_actor(event.get("object1"))
            actor2 = self._house_actor(event.get("object2"))
        else:
            raise ValueError(f"Unexpected swap task {self.task_key!r}")
        count = max(1, int(event.get("count", 1)))
        for _ in range(count):
            self._swap_actors(
                actor1,
                actor2,
                asymmetric=self.task_key in {"bio4", "house3"},
                lift_height=0.2 if self.task_key == "house1" else 0.15,
            )

    def _tube_handoff(self, event: Mapping[str, Any]) -> None:
        if self.task_key == "bio5":
            raise DemonstrationReplayError("bio5 events must use the full source trajectory")
        if self._bio5_planners is not None:
            self._replay_bio5_handoff(event)
            return
        actor = self._tube_actor(event.get("object"))
        self._move_actor(actor, [BIO5_HANDOFF_POSITION], steps_per_target=30)

    def _tube_place(self, event: Mapping[str, Any]) -> None:
        if self.task_key == "bio5":
            raise DemonstrationReplayError("bio5 events must use the full source trajectory")
        if self._bio5_planners is not None:
            if event.get("type") == "distractor_place":
                self._replay_bio5_direct_place(event)
            else:
                self._replay_bio5_place(event)
            return
        actor = self._tube_actor(event.get("object"))
        location = str(event.get("location", "D1"))
        target = self._tube_location(location)
        self._move_actor(
            actor,
            [[target[0], target[1], target[2] + 0.12], target],
            steps_per_target=30,
        )

    def _tube_pass_over(self, event: Mapping[str, Any]) -> None:
        if self.task_key == "bio5":
            raise DemonstrationReplayError("bio5 events must use the full source trajectory")
        if self._bio5_planners is not None:
            self._replay_bio5_pass_over(event)
            return
        # The source fake_place event moves only the right robot.  This
        # scene-only replay preserves its 30+50+50+30 frame duration without
        # creating a tube motion that did not occur in the source scene.
        self._hold(160)

    def _tube_fake_handoff(self, event: Mapping[str, Any]) -> None:
        if self.task_key == "bio5":
            raise DemonstrationReplayError("bio5 events must use the full source trajectory")
        if self._bio5_planners is not None:
            self._replay_bio5_fake_handoff()
            return
        right_arm = None
        agent = getattr(self.raw_env, "agent", None)
        agents = getattr(agent, "agents", None)
        if agents is not None and len(agents) > 1:
            right_arm = agents[1]
        right_tcp = getattr(right_arm, "tcp_pose", None)
        right_position = getattr(right_tcp, "p", None)
        if right_position is not None and float(_to_vector(right_position, 3)[2]) >= 0.2:
            self._hold(300)
        else:
            self._hold(380)

    def _bio5_has_bimanual_agents(self) -> bool:
        agents = getattr(getattr(self.raw_env, "agent", None), "agents", None)
        return agents is not None and len(agents) == 2

    def _initialize_bio5_motion_planners(self) -> None:
        """Set up source-style single-arm planners over the L5 multi-agent env."""
        from mani_skill.examples.motionplanning.double_piper_x.motionplanner import (
            PiperXArmMotionPlanningSolver,
        )

        agents = list(self.raw_env.agent.agents)
        self.raw_env.is_bimanual = True
        self.raw_env.left_arm = agents[0]
        self.raw_env.right_arm = agents[1]
        planners: list[Any] = []
        for index, arm_name in enumerate(("left", "right")):
            planner = PiperXArmMotionPlanningSolver(
                self.raw_env,
                debug=False,
                vis=False,
                base_pose=agents[index].robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                arm=arm_name,
                joint_vel_limits=0.5,
                joint_acc_limits=0.5,
            )
            self._configure_bio5_capture(planner)
            planners.append(planner)
        self._bio5_planners = (planners[0], planners[1])
        for planner, passive_index in ((planners[0], 1), (planners[1], 0)):
            self._bio5_prepare_active(planner, passive_index, gripper_open=True)
            planner.open_gripper(t=8)
            planner.clear_passive_arm_target_qpos()
            planner.clear_passive_gripper_state()

    def _configure_bio5_capture(self, planner: Any) -> None:
        original_step = planner._step_env

        def capture_step(action: Any):
            result = original_step(action)
            self._capture_source_step()
            return result

        planner._step_env = capture_step

    def _bio5_agents(self) -> list[Any]:
        if not self._bio5_has_bimanual_agents():
            raise DemonstrationReplayError("bio5 requires exactly two Piper-X agents")
        return list(self.raw_env.agent.agents)

    def _bio5_prepare_active(
        self, planner: Any, passive_index: int, *, gripper_open: bool
    ) -> None:
        passive_qpos = _to_vector(self._bio5_agents()[passive_index].robot.get_qpos(), None)
        planner.set_passive_arm_target_qpos(passive_qpos)
        planner.set_passive_gripper_state(planner.OPEN if gripper_open else planner.CLOSED)

    def _bio5_move(
        self, planner: Any, position: np.ndarray, quaternion: np.ndarray, *, passive_index: int,
        passive_open: bool,
    ) -> None:
        self._bio5_prepare_active(planner, passive_index, gripper_open=passive_open)
        pose = sapien.Pose(p=np.asarray(position, dtype=np.float32), q=np.asarray(quaternion, dtype=np.float32))
        result = planner.move_to_pose_with_screw(pose)
        if result == -1:
            result = planner.move_to_pose_with_RRTConnect(pose)
        planner.clear_passive_arm_target_qpos()
        planner.clear_passive_gripper_state()
        if result == -1:
            raise DemonstrationReplayError("bio5 source motion plan failed")

    def _bio5_set_gripper(self, planner: Any, *, passive_index: int, open_gripper: bool, passive_open: bool, steps: int = 20) -> None:
        self._bio5_prepare_active(planner, passive_index, gripper_open=passive_open)
        (planner.open_gripper if open_gripper else planner.close_gripper)(t=steps)
        planner.clear_passive_arm_target_qpos()
        planner.clear_passive_gripper_state()

    @staticmethod
    def _bio5_tcp_position(agent: Any) -> np.ndarray:
        return _to_vector(agent.tcp_pose.p, 3)

    def _bio5_grasp_tube(self, planner: Any, arm_index: int, actor: Any) -> np.ndarray:
        tube_position, _ = self._actor_pose(actor)
        approach = tube_position + np.asarray([-0.05, 0.0, BIO5_GRASP_HEIGHT + 0.10], dtype=np.float32)
        grasp = tube_position + np.asarray([0.0, 0.0, BIO5_GRASP_HEIGHT], dtype=np.float32)
        self._bio5_set_gripper(planner, passive_index=1 - arm_index, open_gripper=True, passive_open=True, steps=10)
        self._bio5_move(planner, approach, BIO5_GRASP_QUATERNION, passive_index=1 - arm_index, passive_open=True)
        self._bio5_move(planner, grasp, BIO5_GRASP_QUATERNION, passive_index=1 - arm_index, passive_open=True)
        self._bio5_set_gripper(planner, passive_index=1 - arm_index, open_gripper=False, passive_open=True, steps=30)
        self._bio5_require_grasp(arm_index, actor)
        self._bio5_move(
            planner,
            tube_position + np.asarray([0.0, 0.0, BIO5_LIFT_HEIGHT], dtype=np.float32),
            BIO5_GRASP_QUATERNION,
            passive_index=1 - arm_index,
            passive_open=True,
        )
        return tube_position

    def _bio5_require_grasp(self, arm_index: int, actor: Any) -> None:
        try:
            grasped = bool(_to_vector(self._bio5_agents()[arm_index].is_grasping(actor), None)[0])
        except Exception as exc:
            raise DemonstrationReplayError("bio5 could not verify the physical tube grasp") from exc
        if not grasped:
            raise DemonstrationReplayError("bio5 gripper did not physically grasp the tube")

    def _replay_bio5_handoff(self, event: Mapping[str, Any]) -> None:
        assert self._bio5_planners is not None
        actor = self._tube_actor(event.get("object"))
        left, _ = self._bio5_planners
        self._bio5_grasp_tube(left, 0, actor)
        handoff_tcp = BIO5_HANDOFF_POSITION + np.asarray([0.0, 0.0, 0.10], dtype=np.float32)
        self._bio5_move(left, handoff_tcp, BIO5_GRASP_QUATERNION, passive_index=1, passive_open=True)
        self._bio5_handoff = {"actor": actor, "object": event.get("object"), "quaternion": BIO5_GRASP_QUATERNION.copy()}

    def _replay_bio5_place(self, event: Mapping[str, Any]) -> None:
        if self._bio5_handoff is None:
            raise DemonstrationReplayError("bio5 place event has no preceding handoff")
        if event.get("object") != self._bio5_handoff.get("object"):
            raise DemonstrationReplayError("bio5 place event does not match handed-off tube")
        assert self._bio5_planners is not None
        left, right = self._bio5_planners
        handoff_tcp = BIO5_HANDOFF_POSITION + np.asarray([0.0, 0.0, 0.10], dtype=np.float32)
        receiver_approach = handoff_tcp + np.asarray([0.0, 0.06, 0.0], dtype=np.float32)
        self._bio5_set_gripper(right, passive_index=0, open_gripper=True, passive_open=False, steps=10)
        self._bio5_move(right, receiver_approach, BIO5_GRASP_QUATERNION, passive_index=0, passive_open=False)
        self._bio5_move(right, handoff_tcp, BIO5_GRASP_QUATERNION, passive_index=0, passive_open=False)
        self._bio5_set_gripper(right, passive_index=0, open_gripper=False, passive_open=False, steps=30)
        self._bio5_require_grasp(1, self._bio5_handoff["actor"])
        self._bio5_set_gripper(left, passive_index=1, open_gripper=True, passive_open=False, steps=25)
        target = self._tube_location(str(event.get("location", "D1")))
        above = target + np.asarray([0.0, 0.0, 0.16], dtype=np.float32)
        self._bio5_move(right, above, BIO5_GRASP_QUATERNION, passive_index=0, passive_open=True)
        self._bio5_move(right, target + np.asarray([0.0, 0.0, BIO5_GRASP_HEIGHT], dtype=np.float32), BIO5_GRASP_QUATERNION, passive_index=0, passive_open=True)
        self._bio5_set_gripper(right, passive_index=0, open_gripper=True, passive_open=True, steps=25)
        self._bio5_move(right, above, BIO5_GRASP_QUATERNION, passive_index=0, passive_open=True)
        self._bio5_handoff = None

    def _replay_bio5_direct_place(self, event: Mapping[str, Any]) -> None:
        assert self._bio5_planners is not None
        actor = self._tube_actor(event.get("object"))
        _, right = self._bio5_planners
        self._bio5_grasp_tube(right, 1, actor)
        target = self._tube_location(str(event.get("location", "DD1")))
        above = target + np.asarray([0.0, 0.0, 0.16], dtype=np.float32)
        self._bio5_move(right, above, BIO5_GRASP_QUATERNION, passive_index=0, passive_open=True)
        self._bio5_move(right, target + np.asarray([0.0, 0.0, BIO5_GRASP_HEIGHT], dtype=np.float32), BIO5_GRASP_QUATERNION, passive_index=0, passive_open=True)
        self._bio5_set_gripper(right, passive_index=0, open_gripper=True, passive_open=True, steps=25)
        self._bio5_move(right, above, BIO5_GRASP_QUATERNION, passive_index=0, passive_open=True)

    def _replay_bio5_pass_over(self, event: Mapping[str, Any]) -> None:
        assert self._bio5_planners is not None
        actor = self._tube_actor(event.get("object"))
        _, right = self._bio5_planners
        tube_position, _ = self._actor_pose(actor)
        start = self._bio5_tcp_position(self._bio5_agents()[1])
        hover = tube_position + np.asarray([0.0, 0.0, 0.16], dtype=np.float32)
        self._bio5_set_gripper(right, passive_index=0, open_gripper=True, passive_open=True, steps=8)
        self._bio5_move(right, hover, BIO5_GRASP_QUATERNION, passive_index=0, passive_open=True)
        self._bio5_move(right, start, BIO5_GRASP_QUATERNION, passive_index=0, passive_open=True)

    def _replay_bio5_fake_handoff(self) -> None:
        assert self._bio5_planners is not None
        left, right = self._bio5_planners
        handoff_tcp = BIO5_HANDOFF_POSITION + np.asarray([0.0, 0.0, 0.10], dtype=np.float32)
        self._bio5_set_gripper(left, passive_index=1, open_gripper=True, passive_open=True, steps=8)
        self._bio5_move(left, handoff_tcp, BIO5_GRASP_QUATERNION, passive_index=1, passive_open=True)
        self._bio5_set_gripper(right, passive_index=0, open_gripper=True, passive_open=True, steps=8)
        self._bio5_move(right, handoff_tcp + np.asarray([0.0, 0.06, 0.0], dtype=np.float32), BIO5_GRASP_QUATERNION, passive_index=0, passive_open=True)
        self._bio5_move(right, handoff_tcp, BIO5_GRASP_QUATERNION, passive_index=0, passive_open=True)
        self._bio5_set_gripper(right, passive_index=0, open_gripper=False, passive_open=True, steps=15)
        self._bio5_set_gripper(left, passive_index=1, open_gripper=True, passive_open=False, steps=15)

    def _cover(self, event: Mapping[str, Any]) -> None:
        # Source H1/H3 scripts place the occluder during reset.  The history
        # event is semantic context for the policy, not an animated motion.
        return

    def _house3_place(self, event: Mapping[str, Any]) -> None:
        order = list(event.get("order", []) or [])
        layouts = self._house3_layouts()
        layout_id = int(self.cfg.get("layout_id", 0)) % len(layouts)
        layout = layouts[layout_id]
        identity_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for object_name in order:
            bowl = self._house_actor(object_name)
            bowl_index = _label_index(str(object_name), "O")
            target_row = layout[bowl_index]
            target = np.asarray(
                target_row["pos"] if isinstance(target_row, Mapping) else target_row,
                dtype=np.float32,
            )
            start, _ = self._actor_pose(bowl)
            for frame_index in range(31):
                alpha = frame_index / 30.0
                position = (1.0 - alpha) * start + alpha * target
                self._set_actor_pose(bowl, position, identity_quaternion)
                self._append_demo_frame()

    @staticmethod
    def _house3_layouts() -> list[Any]:
        from mani_skill.envs.tasks.custom_task.household.scene_H3_04 import (
            SceneH3_04Env,
        )

        return SceneH3_04Env.PREDEFINED_BOWL_POSITIONS

    def _remove_occluder(self, event: Mapping[str, Any]) -> None:
        baffle = self._actor("baffle")
        self._move_actor_inclusive(
            baffle,
            np.asarray([0.0, -10.0, 0.0], dtype=np.float32),
            interpolation_steps=50,
            arc_height=0.3,
        )

    def _replay_house5_handoff(self, event: Mapping[str, Any]) -> None:
        """Execute the source House5 history through bimanual control actions.

        The public manifest selects the plate and table target. Replay ends
        after the receiver retracts, before the source script's mode branch.
        """
        plate = self._house_actor(f"O{int(event['object_index']) + 1}")
        table_target = HOUSE5_TABLE_TARGETS[
            int(event["place_index"]) % len(HOUSE5_TABLE_TARGETS)
        ]
        agents = self._house5_agents()
        plate_position, _ = self._actor_pose(plate)
        source_index = min(
            range(2),
            key=lambda index: np.linalg.norm(
                self._house5_tcp_position(agents[index]) - plate_position
            ),
        )
        receiver_index = 1 - source_index
        if self._house5_planners is None:
            raise DemonstrationReplayError("house5 planners were not initialized")
        source = self._house5_planners[source_index]
        receiver = self._house5_planners[receiver_index]
        source_home = self._house5_tcp_pose(agents[source_index])
        receiver_home = self._house5_tcp_pose(agents[receiver_index])

        from scipy.spatial.transform import Rotation

        # Source pass_to(): approach, grasp, lift, and hold at the handoff point.
        # These are the defaults supplied by run_scene_H5_02.py's task-240
        # entrypoint, not the helper's fallback orientation.
        approach = plate_position + np.asarray([-0.15, 0.0, 0.20], dtype=np.float32)
        grasp = plate_position + np.asarray(
            [0.035, 0.0, 0.05 if plate_position[0] < -0.05 else 0.03],
            dtype=np.float32,
        )
        lift = grasp + np.asarray([0.0, -0.10, 0.30], dtype=np.float32)
        # ``pass_to`` holds the plate at target_pos + [0.20, 0.00, 0.25].
        # target_pos is selected by the manifest's place_index.
        handoff = table_target + np.asarray([0.20, 0.0, 0.25], dtype=np.float32)
        source_grasp_quaternion = Rotation.from_euler(
            "xyz", [-90.0, 0.0, 180.0], degrees=True
        ).as_quat().astype(np.float32)
        source_handoff_quaternion = Rotation.from_euler(
            "xyz", [-117.0, 56.0, -185.0], degrees=True
        ).as_quat().astype(np.float32)
        self._house5_prepare_active(source, receiver_index, gripper_open=False)
        # ``pass_to`` opens the source gripper through its first motion action;
        # it does not add a separate opening interval before that motion.
        source.gripper_state = source.OPEN
        self._house5_move(
            source, approach, source_grasp_quaternion, limit_search_range=True
        )
        self._house5_move(
            source, grasp, source_grasp_quaternion, limit_search_range=True
        )
        self._house5_prepare_active(source, receiver_index, gripper_open=False)
        self._set_house5_gripper_drive(agents[source_index], grasping=True)
        source.close_gripper(t=50)
        self._house5_move(
            source, lift, source_grasp_quaternion, limit_search_range=True
        )
        self._house5_move(
            source,
            lift + np.asarray([0.0, -0.10, 0.0], dtype=np.float32),
            source_grasp_quaternion,
            limit_search_range=False,
        )
        self._house5_move(
            source,
            handoff,
            source_handoff_quaternion,
            limit_search_range=True,
        )

        # Source handoff: receiver approaches, closes, source releases, then retracts.
        # The collector first rotates the plate about world Z to make the
        # handoff orientation perpendicular to the receiver's fixed grasp.
        _, plate_quaternion = self._actor_pose(plate)
        plate_yaw = Rotation.from_quat(plate_quaternion).as_euler(
            "zyx", degrees=True
        )[0]
        plate_yaw = (plate_yaw + 180.0) % 360.0 - 180.0
        if abs(plate_yaw) > 1.0:
            source_pose = self._house5_tcp_pose(agents[source_index])
            adjusted_source_quaternion = (
                Rotation.from_euler("z", plate_yaw, degrees=True)
                * Rotation.from_quat(_to_vector(source_pose.q, 4))
            ).as_quat().astype(np.float32)
            self._house5_move(
                source,
                _to_vector(source_pose.p, 3),
                adjusted_source_quaternion,
                limit_search_range=True,
            )

        source_tcp = self._house5_tcp_position(agents[source_index])
        plate_after_lift, _ = self._actor_pose(plate)
        receiver_pre_handoff = source_tcp + np.asarray(
            [-0.07, 0.0, -0.03], dtype=np.float32
        )
        receiver_pre_handoff[1] = plate_after_lift[1] - 0.08
        receiver_handoff = source_tcp + np.asarray(
            [0.03, 0.0, -0.03], dtype=np.float32
        )
        receiver_handoff[1] = plate_after_lift[1] - 0.08
        receiver_quaternion = self._house5_receiver_handoff_quaternion()
        self._house5_prepare_active(receiver, source_index, gripper_open=False)
        # The source's two receiver approach motions use an open gripper.
        receiver.gripper_state = receiver.OPEN
        receiver_approached = self._house5_move(
            receiver,
            receiver_pre_handoff,
            receiver_quaternion,
            limit_search_range=True,
        )
        # run_scene_H5_02.py holds both arms for ten recorded control steps
        # after each receiver approach to stabilize the plate before closing.
        self._house5_prepare_active(source, receiver_index, gripper_open=True)
        self._house5_hold_active(source, steps=10)
        if receiver_approached:
            receiver_approached = self._house5_move(
                receiver,
                receiver_handoff,
                receiver_quaternion,
                limit_search_range=True,
            )
            if receiver_approached:
                self._house5_prepare_active(source, receiver_index, gripper_open=True)
                self._house5_hold_active(source, steps=10)
        self._house5_prepare_active(receiver, source_index, gripper_open=False)
        self._set_house5_gripper_drive(agents[receiver_index], grasping=True)
        receiver.close_gripper(t=50)
        self._house5_prepare_active(source, receiver_index, gripper_open=False)
        source.open_gripper(t=30)
        source_after_release = self._house5_tcp_pose(agents[source_index])
        self._house5_move(
            source,
            _to_vector(source_after_release.p, 3)
            + np.asarray([-0.10, 0.0, 0.0], dtype=np.float32),
            _to_vector(source_after_release.q, 4),
            limit_search_range=True,
        )
        self._house5_move(
            source,
            _to_vector(source_home.p, 3)
            + np.asarray([0.0, 0.0, 0.05], dtype=np.float32),
            _to_vector(source_home.q, 4),
            limit_search_range=False,
        )

        # The last history labels are "right arm places the plate (on table)"
        # and "right arm retracts". Do not execute the following mode branch.
        table_quaternion = Rotation.from_euler(
            "xyz", [-135.0, 0.0, -135.0], degrees=True
        ).as_quat()
        self._house5_prepare_active(receiver, source_index, gripper_open=True)
        self._house5_move(
            receiver, table_target, table_quaternion, limit_search_range=False
        )
        self._house5_prepare_active(receiver, source_index, gripper_open=True)
        receiver.open_gripper(t=30)
        receiver_after_release = self._house5_tcp_pose(agents[receiver_index])
        self._house5_move(
            receiver,
            _to_vector(receiver_after_release.p, 3)
            + np.asarray([-0.20, 0.0, 0.10], dtype=np.float32),
            _to_vector(receiver_after_release.q, 4),
            limit_search_range=False,
        )
        self._house5_move(
            receiver,
            _to_vector(receiver_home.p, 3)
            + np.asarray([0.0, 0.0, 0.05], dtype=np.float32),
            _to_vector(receiver_home.q, 4),
            limit_search_range=False,
        )

    def _house5_agents(self) -> list[Any]:
        agents = getattr(getattr(self.raw_env, "agent", None), "agents", None)
        if agents is None or len(agents) != 2:
            raise DemonstrationReplayError("house5 requires exactly two Piper-X agents")
        return list(agents)

    def _initialize_house5_motion_planners(self) -> None:
        from mani_skill.examples.motionplanning.double_piper_x.motionplanner import (
            PiperXArmMotionPlanningSolver,
        )

        env = self.raw_env
        agents = self._house5_agents()
        # SceneH5 builds a MultiAgent directly, while the reusable Piper
        # solver expects the aliases published by the industrial bimanual env.
        env.is_bimanual = True
        env.left_arm = agents[0]
        env.right_arm = agents[1]
        self._configure_house5_drives(agents)
        self._configure_house5_grasp_physics(agents)
        planners = []
        for index, arm_name in enumerate(("left", "right")):
            planner = PiperXArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=agents[index].robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                arm=arm_name,
                joint_vel_limits=0.5,
                joint_acc_limits=0.5,
            )
            self._configure_ind5_capture(planner)
            planners.append(planner)
        self._house5_planners = (planners[0], planners[1])

        # The source collector first settles the scene for 50 closed-gripper
        # control steps, then records ten identical initial frames.
        for _ in range(50):
            self._house5_step_held_arms(capture=False)
        for _ in range(10):
            self._house5_step_held_arms(capture=True)

    @staticmethod
    def _house5_qpos(agent: Any) -> np.ndarray:
        return _to_vector(agent.robot.get_qpos(), None)

    @staticmethod
    def _house5_tcp_position(agent: Any) -> np.ndarray:
        return _to_vector(agent.tcp_pose.p, 3)

    @staticmethod
    def _house5_tcp_pose(agent: Any) -> sapien.Pose:
        pose = agent.tcp_pose
        return sapien.Pose(p=_to_vector(pose.p, 3), q=_to_vector(pose.q, 4))

    @staticmethod
    def _house5_receiver_handoff_quaternion() -> np.ndarray:
        # Same frame construction as ``run_scene_H5_02.py`` step 1a.
        from scipy.spatial.transform import Rotation

        rotation = np.column_stack(
            (
                np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        return Rotation.from_matrix(rotation).as_quat().astype(np.float32)

    def _house5_prepare_active(
        self, planner: Any, passive_index: int, *, gripper_open: bool
    ) -> None:
        passive_qpos = self._house5_qpos(self._house5_agents()[passive_index])
        planner.set_passive_arm_target_qpos(passive_qpos)
        planner.set_passive_gripper_state(planner.OPEN if gripper_open else planner.CLOSED)

    @staticmethod
    def _configure_house5_drives(agents: list[Any]) -> None:
        """Match the collector's high-stiffness hold used during plate handoff."""
        for agent in agents:
            for joint in agent.robot.get_active_joints():
                if "joint7" in joint.name or "joint8" in joint.name:
                    joint.set_drive_properties(
                        stiffness=100_000,
                        damping=100_000,
                        force_limit=50_000,
                    )
                else:
                    joint.set_drive_properties(
                        stiffness=1_000_000,
                        damping=100_000,
                        force_limit=1_000_000,
                    )

    def _configure_house5_grasp_physics(self, agents: list[Any]) -> None:
        """Apply the collector's plate and fingertip contact parameters."""
        for agent in agents:
            try:
                for link in agent.robot.get_links():
                    for shape in link.get_collision_shapes():
                        material = shape.physical_material
                        material.static_friction = 2.0
                        material.dynamic_friction = 2.0
                        material.restitution = 0.0
            except Exception:
                # This matches the collector's best-effort physics tuning and
                # keeps a renderer/backend-specific shape from aborting reset.
                continue

        for name in ("plate1", "plate2", "plate3"):
            plate = getattr(self.raw_env, name, None)
            if plate is None:
                continue
            try:
                for link in plate.get_links():
                    link.mass = 0.01
                    for shape in link.get_collision_shapes():
                        material = shape.physical_material
                        material.static_friction = 1.0
                        material.dynamic_friction = 1.0
                        material.restitution = 0.01
            except Exception:
                continue

    @staticmethod
    def _set_house5_gripper_drive(agent: Any, *, grasping: bool) -> None:
        """Use the collector's temporary drive values while closing a plate."""
        if grasping:
            stiffness, damping, force_limit = 10_000, 500, 1_000
        else:
            stiffness, damping, force_limit = 100_000, 100_000, 50_000
        for joint in agent.robot.get_active_joints():
            if "joint7" in joint.name or "joint8" in joint.name:
                joint.set_drive_properties(
                    stiffness=stiffness,
                    damping=damping,
                    force_limit=force_limit,
                )

    def _house5_move(
        self,
        planner: Any,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        limit_search_range: bool,
    ) -> bool:
        """Execute a source-style adaptive motion plan without aborting reset.

        ``run_scene_H5_02.py`` does not treat a fixed TCP pose as the only
        valid solution.  It searches nearby orientations and positions before
        declaring a segment unreachable.  The earlier replay replaced that
        behavior with one screw plan and one RRT plan, so an ordinary IK
        miss prevented the policy from receiving any post-demo steps.
        """
        target_position = np.asarray(position, dtype=np.float32)
        target_quaternion = np.asarray(quaternion, dtype=np.float32)
        # Keep House5's source planning contract: exact screw planning first,
        # then the collector's adaptive pose search.  In particular, do not
        # substitute an RRT solution before the adaptive search; the source
        # collector uses the latter to preserve a plate-compatible grasp.
        result = self._house5_plan_screw(
            planner, target_position, target_quaternion
        )
        if result is not None:
            planner.follow_path(result)
            return True

        for candidate_position, candidate_quaternion in self._house5_pose_candidates(
            planner,
            target_position,
            target_quaternion,
            limit_search_range=limit_search_range,
        ):
            result = self._house5_plan_screw(
                planner, candidate_position, candidate_quaternion
            )
            if result is not None:
                planner.follow_path(result)
                return True

        # A replay error must not prevent the benchmark's policy phase.  Keep
        # both arms commanded at their current positions for a source-like
        # control interval, then let reset return normally.
        self._house5_hold_active(planner, steps=30)
        self._house5_replay_warnings.append(
            "house5 motion segment was unreachable; held current dual-arm pose"
        )
        return False

    @staticmethod
    def _house5_plan_screw(
        planner: Any, position: np.ndarray, quaternion: np.ndarray
    ) -> Any | None:
        joint_dim = len(planner.planner.joint_vel_limits)
        current_qpos = _to_vector(planner.robot.get_qpos(), None)[:joint_dim]
        try:
            result = planner.planner.plan_screw(
                np.concatenate([position, quaternion]),
                current_qpos,
                time_step=planner.base_env.control_timestep,
                use_point_cloud=False,
            )
        except Exception:
            return None
        return result if result.get("status") == "Success" else None

    def _house5_pose_candidates(
        self,
        planner: Any,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        *,
        limit_search_range: bool,
    ):
        """Yield candidates in source ``find_reachable_pose`` priority order."""
        from scipy.spatial.transform import Rotation, Slerp

        deadline = time.monotonic() + 5.0
        current_quaternion = _to_vector(planner.active_arm.tcp_pose.q, 4)
        if limit_search_range:
            rotations = Rotation.from_quat([current_quaternion, target_quaternion])
            # The source checks all fractions then returns the largest
            # reachable one. Yielding from 1.0 downward gives that same final
            # pose while avoiding a lower-fidelity, nearly-current orientation
            # being accepted before the intended plate grasp.
            for fraction in np.linspace(1.0, 0.1, 10):
                if time.monotonic() > deadline:
                    return
                yield target_position, Slerp([0.0, 1.0], rotations)(fraction).as_quat()

        if limit_search_range:
            roll_candidates = (0.0, np.pi / 6, -np.pi / 6, np.pi / 4, -np.pi / 4, np.pi / 3, -np.pi / 3)
            pitch_candidates = roll_candidates
            yaw_candidates = (0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2)
        else:
            roll_candidates = (0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2, 3 * np.pi / 4, -3 * np.pi / 4, np.pi)
            pitch_candidates = roll_candidates
            yaw_candidates = (0.0, np.pi / 2, -np.pi / 2, np.pi)
        rotation_offsets = tuple(
            Rotation.from_euler("xyz", offset).as_quat()
            for offset in product(roll_candidates, pitch_candidates, yaw_candidates)
        )

        # Source utility: 32 samples at a 3 cm radius, with position as the
        # outer loop and orientation offset as the inner loop. Its quaternion
        # composition is offset * target, not target * offset.
        samples = 32
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for index in range(samples):
            if time.monotonic() > deadline:
                return
            y = 1.0 - (index / float(samples - 1)) * 2.0
            radius = np.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * index
            offset = np.asarray(
                [np.cos(theta) * radius, y, np.sin(theta) * radius],
                dtype=np.float32,
            ) * 0.03
            for offset_quaternion in rotation_offsets:
                if time.monotonic() > deadline:
                    return
                candidate_quaternion = (
                    Rotation.from_quat(offset_quaternion)
                    * Rotation.from_quat(target_quaternion)
                ).as_quat().astype(np.float32)
                yield target_position + offset, candidate_quaternion

        # Last source fallback: an expanded 9 cm positional search that keeps
        # the intended grasp orientation unchanged.
        for index in range(20):
            if time.monotonic() > deadline:
                return
            y = 1.0 - (index / 19.0) * 2.0
            radius = np.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * index
            offset = np.asarray(
                [np.cos(theta) * radius, y, np.sin(theta) * radius],
                dtype=np.float32,
            ) * 0.09
            yield target_position + offset, target_quaternion

    @staticmethod
    def _house5_hold_active(planner: Any, *, steps: int) -> None:
        joint_dim = len(planner.planner.joint_vel_limits)
        qpos = _to_vector(planner.robot.get_qpos(), None)[:joint_dim]
        for _ in range(steps):
            active_action = np.hstack([qpos, planner.gripper_state])
            planner._step_env(planner._build_dual_arm_action(active_action))

    def _house5_step_held_arms(self, *, capture: bool) -> None:
        action = {
            f"piper_x-{index}": np.hstack([self._house5_qpos(agent)[:6], -1.0])
            for index, agent in enumerate(self._house5_agents())
        }
        self.raw_env.step(action)
        if capture:
            self._capture_source_step()

    def _circle_actor(
        self,
        actor: Any,
        *,
        center: np.ndarray,
        radius: float,
        circles: int,
        lift_after_each: bool = False,
    ) -> None:
        _, quat = self._actor_pose(actor)
        for circle in range(max(1, circles)):
            for frame_index in range(31):
                angle = 2.0 * np.pi * frame_index / 30.0
                position = center + np.asarray(
                    [radius * np.cos(angle), radius * np.sin(angle), 0.0],
                    dtype=np.float32,
                )
                self._set_actor_pose(actor, position, quat)
                self._append_demo_frame()
            if lift_after_each:
                current, _ = self._actor_pose(actor)
                self._move_actor_inclusive(
                    actor,
                    current + np.asarray([0.0, 0.0, 0.15], dtype=np.float32),
                    interpolation_steps=20,
                )
                if circle < circles - 1:
                    lowered, _ = self._actor_pose(actor)
                    lowered[2] = center[2]
                    self._move_actor_inclusive(
                        actor, lowered, interpolation_steps=20
                    )

    def _move_actor_inclusive_with_quaternion(
        self,
        actor: Any,
        target: np.ndarray,
        quaternion: np.ndarray,
        *,
        interpolation_steps: int,
    ) -> None:
        start, _ = self._actor_pose(actor)
        target = np.asarray(target, dtype=np.float32)
        for frame_index in range(interpolation_steps + 1):
            alpha = frame_index / interpolation_steps
            position = (1.0 - alpha) * start + alpha * target
            self._set_actor_pose(actor, position, quaternion)
            self._append_demo_frame()

    def _return_bio4_rod_to_initial(self) -> None:
        """Hide the rod above the work area before returning it to storage."""
        rod = self._actor("glass_rod1")
        _, quaternion = self._actor_pose(rod)
        current, _ = self._actor_pose(rod)
        hidden_above_work_area = current.copy()
        hidden_above_work_area[2] = BIO4_ROD_HIDE_HEIGHT
        hidden_above_storage = BIO4_ROD_HOME_POSITION.copy()
        hidden_above_storage[2] = BIO4_ROD_HIDE_HEIGHT

        self._move_actor_inclusive_with_quaternion(
            rod,
            hidden_above_work_area,
            quaternion,
            interpolation_steps=30,
        )
        self._move_actor_inclusive_with_quaternion(
            rod,
            hidden_above_storage,
            quaternion,
            interpolation_steps=30,
        )
        self._move_actor_inclusive_with_quaternion(
            rod,
            BIO4_ROD_HOME_POSITION,
            quaternion,
            interpolation_steps=30,
        )

    def _animate_qpos(
        self, articulation: Any, index: int, start: float, end: float, *, steps: int
    ) -> None:
        initial = _to_vector(articulation.get_qpos(), None)
        for frame_index in range(steps):
            alpha = (frame_index + 1) / steps
            qpos = initial.copy()
            qpos[index] = (1.0 - alpha) * start + alpha * end
            articulation.set_qpos(qpos)
            self._append_demo_frame()

    def _animate_ind2_lid(
        self, articulation: Any, index: int, *, target_angle: float
    ) -> None:
        """Match ``collect_skipped_bin.py`` lid motion and capture timing."""
        steps = 15
        joint = articulation.active_joints[index]
        joint.set_drive_properties(
            stiffness=100, damping=10, force_limit=100, mode="force"
        )
        qpos = _to_vector(articulation.get_qpos(), None)
        start_angle = float(qpos[index])
        for frame_index in range(steps):
            alpha = (frame_index + 1) / steps
            qpos[index] = (1.0 - alpha) * start_angle + alpha * target_angle
            articulation.set_qpos(qpos)
            joint.set_drive_target(float(qpos[index]))
            self.raw_env.scene.step()
            self._append_demo_frame()

    def _hold(self, frames: int) -> None:
        for _ in range(frames):
            self._append_demo_frame()

    def _step_scene_and_capture(self, frames: int) -> None:
        """Match source history segments that advance physics before rendering."""
        for _ in range(frames):
            self.raw_env.scene.step()
            self._append_demo_frame()

    def _lid_joint_index(self, articulation: Any) -> int | None:
        for index, joint in enumerate(getattr(articulation, "active_joints", [])):
            if "lid_joint" in getattr(joint, "name", ""):
                return index
        return None

    def _tube_actor(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("T"):
            return self._actor(f"sample_tube_{_label_index(value, 'T')}")
        return self._actor(value)

    def _tube_location(self, location: str) -> np.ndarray:
        if location.startswith("DD"):
            index = _label_index(location[1:], "D")
            return BIO5_DD_RACK_POSITIONS[index % len(BIO5_DD_RACK_POSITIONS)].copy()
        index = _label_index(location, "D")
        return BIO5_D_RACK_POSITIONS[index % len(BIO5_D_RACK_POSITIONS)].copy()

    def _house_actor(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("O"):
            prefix = "plate" if self.task_key in {"house1", "house5"} else "bowl"
            return self._actor(f"{prefix}{_label_index(value, 'O') + 1}")
        return self._actor(value)


def primary_rgb_from_obs(
    obs: Mapping[str, Any], *, preferred_camera: str | None = None
) -> tuple[str, np.ndarray]:
    sensor_data = obs.get("sensor_data")
    if not isinstance(sensor_data, Mapping):
        raise ValueError("Observation does not contain sensor_data")
    ordered = [
        *([preferred_camera] if preferred_camera is not None else []),
        *PRIMARY_CAMERA_ORDER,
        *(name for name in sensor_data if name not in PRIMARY_CAMERA_ORDER),
    ]
    for name in ordered:
        camera = sensor_data.get(name)
        if not isinstance(camera, Mapping):
            continue
        for key in ("rgb", "Color", "color"):
            if key in camera:
                return name, _to_uint8_rgb(camera[key])
    raise ValueError(f"No RGB sensor image found in cameras={list(sensor_data)}")



def _label_index(value: str, prefix: str) -> int:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"Expected {prefix}-label, got {value!r}")
    return int(value[len(prefix) :]) - 1


def _to_vector(value: Any, length: int | None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim > 1:
        vector = vector.reshape(-1, vector.shape[-1])[0]
    vector = vector.reshape(-1)
    if length is not None and vector.size < length:
        raise ValueError(f"Expected vector length {length}, got {vector.size}")
    return vector[:length].copy() if length is not None else vector.copy()


def _to_uint8_rgb(value: Any) -> np.ndarray:
    array = _to_vector_image(value)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] < 3:
        raise ValueError(f"RGB image must have at least 3 channels, got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            high = float(array.max(initial=0))
            array = np.clip(array, 0.0, 1.0) * 255.0 if high <= 1.0 else np.clip(array, 0, 255)
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array[..., :3])


def _to_vector_image(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Image must be HWC or NHWC, got shape={array.shape}")
    return array
