"""Gym/ManiSkill environment adapter for GITBench."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .paths import ensure_maniskill_on_path

# SAPIEN may live in a separate ManiSkill runtime environment.  Configure its
# optional site-packages path before importing SAPIEN directly or indirectly.
ensure_maniskill_on_path()

import numpy as np
import sapien
import torch

from .episodes import (
    episode_reset_config,
    episode_seed,
    evaluator_config,
    get_episode,
    public_episode_info,
)
from .demonstrations import (
    DemonstrationReplayError,
    SceneDemonstrationReplayer,
    normalize_history_events,
)
from .policy import zero_action
from .progress import normalize_progress
from .tasks import TaskSpec, get_task, import_all_task_envs


FRONT_RENDER_CAMERA_EYE = [0.6, 0.0, 0.6]
FRONT_RENDER_CAMERA_TARGET = [0.0, 0.0, 0.25]
FRONT_RENDER_CAMERA_FOV = 1.0
FRONT_RENDER_CAMERA_WIDTH = 1280
FRONT_RENDER_CAMERA_HEIGHT = 720


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _to_bool(value: Any) -> bool:
    arr = _to_numpy(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(arr.reshape(-1)[0])


def _jsonable(value: Any) -> Any:
    if callable(value):
        return getattr(value, "__name__", repr(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach") or hasattr(value, "numpy"):
        arr = _to_numpy(value)
        return arr.item() if arr.shape == () else arr.tolist()
    return repr(value)


def _unwrap(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        nxt = getattr(current, "env", None)
        if nxt is not None and nxt is not current:
            current = nxt
            continue
        nxt = getattr(current, "unwrapped", None)
        if nxt is not None and nxt is not current:
            current = nxt
            continue
        break
    return current


def _front_render_camera_configs(shader: str) -> dict[str, Any]:
    from mani_skill.utils import sapien_utils

    pose = sapien_utils.look_at(
        eye=FRONT_RENDER_CAMERA_EYE,
        target=FRONT_RENDER_CAMERA_TARGET,
    )
    return {
        "shader_pack": shader,
        "render_camera": {
            "pose": pose,
            "width": FRONT_RENDER_CAMERA_WIDTH,
            "height": FRONT_RENDER_CAMERA_HEIGHT,
            "fov": FRONT_RENDER_CAMERA_FOV,
            "intrinsic": None,
            "near": 0.01,
            "far": 100,
            "shader_pack": shader,
        },
    }


class GITBenchEnv:
    """Small benchmark wrapper around the local HI-VLA evaluation tasks."""

    def __init__(
        self,
        task: str | TaskSpec,
        *,
        obs_mode: str = "rgbd",
        control_mode: str = "pd_joint_pos",
        render_mode: str | None = "rgb_array",
        shader: str = "default",
        sim_backend: str = "auto",
        sim_freq: int = 120,
        control_freq: int = 30,
        max_episode_steps: int | None = None,
        success_mode: str = "ever",
        terminate_on_success: bool = False,
        make_env: bool = True,
    ) -> None:
        self.spec = get_task(task) if isinstance(task, str) else task
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.shader = shader
        self.sim_backend = sim_backend
        self.sim_config = {"sim_freq": sim_freq, "control_freq": control_freq}
        self.max_episode_steps = int(
            self.spec.max_steps if max_episode_steps is None else max_episode_steps
        )
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if success_mode not in {"ever", "final"}:
            raise ValueError("success_mode must be 'ever' or 'final'")
        self.success_mode = success_mode
        self.terminate_on_success = bool(terminate_on_success)
        self.env = None
        self.raw_env = None
        self.episode_config: dict[str, Any] = {}
        self._public_episode: dict[str, Any] = {}
        self._evaluator_config: dict[str, Any] = dict(self.spec.default_evaluator)
        self._episode_success_targets: dict[str, np.ndarray] = {}
        self._failure_baseline_positions: dict[str, np.ndarray] = {}
        self._failure_baseline_states: dict[tuple[int, str], bool] = {}
        self._success_latched = False
        self._failure_latched = False
        self._failure_reason: dict[str, Any] | None = None
        self._progress_config: dict[str, Any] = normalize_progress(
            None, self._evaluator_config, task_key=self.spec.key
        )
        self._progress_baseline_positions: dict[str, np.ndarray] = {}
        self._progress_milestones: list[dict[str, Any]] = []
        self._progress_stable_counts: dict[str, int] = {}
        self._progress_step = 0
        self._progress_trace: list[dict[str, Any]] = []
        self._progress_completion = 0.0
        self.last_demonstration: dict[str, Any] | None = None
        self._demonstration_frame_callback: Callable[[bool], None] | None = None
        self.last_obs = None
        self.last_info: dict[str, Any] = {}
        if make_env:
            self.make()

    @property
    def action_space(self):
        if self.env is None:
            self.make()
        return self.env.action_space

    @property
    def observation_space(self):
        if self.env is None:
            self.make()
        return self.env.observation_space

    def make(self, make_config: Mapping[str, Any] | None = None):
        ensure_maniskill_on_path()
        import_all_task_envs()
        import gymnasium as gym

        kwargs = self._gym_make_kwargs(make_config)
        self.env = gym.make(self.spec.gym_env_id, **kwargs)
        self.raw_env = _unwrap(self.env)
        return self.env

    def _gym_make_kwargs(self, make_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        cfg = dict(self.spec.default_episode)
        if make_config:
            cfg.update(make_config)
        kwargs: dict[str, Any] = {
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "render_mode": self.render_mode,
            "sensor_configs": {"shader_pack": self.shader},
            "human_render_camera_configs": _front_render_camera_configs(self.shader),
            "viewer_camera_configs": {"shader_pack": self.shader},
            "sim_backend": self.sim_backend,
            "robot_uids": ("piper_x", "piper_x"),
            "sim_config": self.sim_config,
            # The benchmark driver owns the episode budget. Source tasks use
            # inconsistent 50/200-step registrations, so override them here.
            "max_episode_steps": self.max_episode_steps,
        }

        if self.spec.key == "bio4":
            kwargs["beaker_position_index"] = cfg.get("beaker_position_index")
        elif self.spec.key == "ind5":
            layout = cfg.get("layout_config", {})
            parts = layout.get("parts", [])
            part_types = layout.get("part_types", {})
            kwargs["num_parts"] = len(parts) or 3
            kwargs["part_types"] = [part_types.get(part, "cube") for part in parts] or None
        elif self.spec.key == "house5":
            kwargs["layout_id"] = cfg.get("layout_id", 0)
        return kwargs

    def reset(
        self,
        *,
        episode_id: int | None = None,
        split: str = "test",
        task_config: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ):
        episode = None
        if task_config is None:
            selected_id = 0 if episode_id is None else episode_id
            episode = get_episode(self.spec.key, selected_id, split=split)
            cfg = self._episode_replay_config(episode)
            self._public_episode = public_episode_info(episode)
            self._evaluator_config = evaluator_config(episode, task_key=self.spec.key)
        else:
            cfg = self.spec.episode_config(task_config)
            if episode_id is not None:
                cfg["episode_id"] = episode_id
            self._public_episode = {
                "episode_id": int(cfg.get("episode_id", 0)),
                "split": "custom",
                "instruction": cfg.get("instruction", ""),
                "history_events": [],
            }
            self._evaluator_config = dict(self.spec.default_evaluator)

        self._progress_config = normalize_progress(
            self._evaluator_config.get("progress"),
            self._evaluator_config,
            task_key=self.spec.key,
        )

        if self._requires_rebuild(cfg):
            self.close()
        if self.env is None:
            self.make(cfg)
        reset_seed = seed if seed is not None else (
            episode_seed(episode) if episode is not None else int(cfg.get("episode_id", 0))
        )

        self._prepare_reset(cfg)
        reset_options = self._reset_options(cfg)
        obs, info = self.env.reset(seed=reset_seed, options=reset_options)
        self.raw_env = _unwrap(self.env)
        self.episode_config = cfg
        self._public_episode["history_events"] = normalize_history_events(
            self.spec.key,
            cfg,
            self._public_episode.get("history_events", []),
        )
        if self.spec.key == "ind2":
            # The source collector closes both grippers for 40 control steps
            # immediately after reset, before applying its episode layout.
            self._stabilize_custom_scene()
            self._apply_episode_config(cfg)
        else:
            self._apply_episode_config(cfg)
            self._stabilize_custom_scene()
        self._capture_episode_success_targets()
        demonstration = self._replay_demonstration()
        self._capture_failure_baselines()
        self._initialize_progress()
        # A new episode always starts with a clean benchmark outcome.  The
        # initial scene is evaluated below and may latch success for trivial
        # (already-satisfied) custom episodes.
        self._success_latched = False
        self._failure_latched = False
        self._failure_reason = None
        self._reset_wrapper_elapsed_steps()
        info = self._policy_info()
        obs = self._fresh_obs(obs)
        self._assert_interaction_frame_matches_observation(demonstration, obs)
        info["demonstration"] = demonstration
        self.last_obs = obs
        self.last_info = info
        return obs, info

    def _episode_replay_config(self, episode: Mapping[str, Any]) -> dict[str, Any]:
        cfg = self.spec.episode_config(episode_reset_config(episode))
        # Source-only provenance selects the exact collector program for
        # replay. It remains internal to reset and is never exposed in
        # policy-facing episode information.
        source = episode.get("source", {})
        if self.spec.key == "bio5" and isinstance(source, Mapping):
            record_id = source.get("record_id")
            if isinstance(record_id, int):
                cfg["source_record_id"] = record_id
        return cfg

    def step(self, action: Any):
        obs, reward, _source_terminated, truncated, _source_info = self.env.step(action)
        current_success = self._benchmark_success_current()
        # Always evaluate failures, including on a step that also satisfies
        # the success geometry. A safety violation wins ties.
        failure = self._benchmark_failure()
        if failure:
            self._failure_latched = True
        elif current_success:
            self._success_latched = True
        self._progress_step += 1
        if not failure:
            if current_success:
                self._force_progress_success()
            else:
                self._update_progress()
        # Recompute after updating the latches so a failure on the same step
        # cannot be reported as success merely because an earlier step had
        # satisfied the target.
        success = self._benchmark_success()
        info = self._policy_info()
        self.last_obs = obs
        self.last_info = info
        # Source task success flags are not comparable across the two task
        # families (some source environments report success unconditionally).
        # GITBench exposes one evaluator instead and lets the evaluation
        # driver decide whether to stop on that public benchmark success.
        # Native ManiSkill ``terminated`` values are deliberately ignored:
        # several source tasks report success unconditionally.  The wrapper's
        # termination contract is benchmark failure plus the optional,
        # explicit terminate-on-success switch.
        terminated = bool(failure or (self.terminate_on_success and success))
        return obs, reward, terminated, truncated, info

    def evaluate(self) -> dict[str, Any]:
        return self._benchmark_info()

    def sample_zero_action(self):
        return zero_action(self.action_space)

    def render(self):
        return self.env.render()

    def set_demonstration_frame_callback(
        self, callback: Callable[[bool], None] | None
    ) -> None:
        """Set a video-only callback invoked at every replayed scene frame.

        ``True`` denotes a demonstration frame; the final callback from
        ``reset()`` is the first normal interaction frame and receives
        ``False``. The callback never affects policy observations.
        """
        self._demonstration_frame_callback = callback

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self.raw_env = None

    @property
    def failure_reason(self) -> dict[str, Any] | None:
        """Private evaluator diagnostic; never included in policy-facing info."""
        return None if self._failure_reason is None else dict(self._failure_reason)

    def _benchmark_info(self) -> dict[str, Any]:
        benchmark_success = self._benchmark_success()
        current_success = self._benchmark_success_current()
        info = {
            "task_key": self.spec.key,
            "env_id": self.spec.env_id,
            "gym_env_id": self.spec.gym_env_id,
            "source_family": self.spec.source_family,
            "episode": _jsonable(self._public_episode),
        }

        info["success"] = bool(benchmark_success) if self.spec.scorable else False
        info["success_current"] = bool(current_success) if self.spec.scorable else False
        info["success_mode"] = self.success_mode
        info["success_rule"] = self.spec.success_rule
        info["scorable"] = self.spec.scorable
        info["progress"] = self._progress_info()
        return info

    def _policy_info(self) -> dict[str, Any]:
        """Return only benchmark-approved context to a policy.

        ManiSkill task implementations may place target names or target poses in
        their native ``info`` mapping. Those source-specific values must not be
        exposed during benchmark evaluation.
        """
        evaluator_info = self._benchmark_info()
        return {
            key: evaluator_info[key]
            for key in (
                "task_key",
                "env_id",
                "gym_env_id",
                "source_family",
                "episode",
                "success",
                "success_current",
                "success_mode",
                "scorable",
            )
            if key in evaluator_info
        }

    def _initialize_progress(self) -> None:
        """Capture policy-start baselines after all scene history replay."""
        self._progress_baseline_positions = {}
        self._progress_milestones = []
        self._progress_stable_counts = {}
        self._progress_step = 0
        self._progress_trace = []
        self._progress_completion = 0.0
        for milestone in self._progress_config.get("milestones", []):
            item = dict(milestone)
            item["achieved"] = False
            item["first_step"] = None
            item["forced_by_success"] = False
            self._progress_milestones.append(item)
            self._progress_stable_counts[item["id"]] = 0
            ref = item.get("params", {}).get("object_ref")
            actor = self._resolve_actor(ref)
            position = self._actor_position(actor)
            if isinstance(ref, str) and position is not None:
                self._progress_baseline_positions.setdefault(ref, position.copy())

    def _progress_rule_observed(self, milestone: Mapping[str, Any]) -> bool:
        rule = milestone.get("rule")
        params = milestone.get("params", {})
        ref = params.get("object_ref")
        if rule == "object_touched":
            return self._object_touched(
                self._resolve_actor(ref),
                min_force=float(params.get("min_force", 0.05)),
            )
        if rule == "object_grasped":
            return self._grasp_state(self._resolve_actor(ref)) is True
        if rule == "object_displaced_from_reset":
            actor = self._resolve_actor(ref)
            current = self._actor_position(actor)
            initial = self._progress_baseline_positions.get(ref)
            if current is None or initial is None:
                return False
            dimensions = 2 if params.get("metric", "xyz") == "xy" else 3
            return bool(np.linalg.norm(current[:dimensions] - initial[:dimensions]) >= float(params.get("threshold", 0.03)))
        if rule == "object_lifted_from_reset":
            actor = self._resolve_actor(ref)
            current = self._actor_position(actor)
            initial = self._progress_baseline_positions.get(ref)
            return bool(current is not None and initial is not None and current[2] - initial[2] >= float(params.get("height", 0.03)))
        if rule == "object_in_rectangle":
            return self._object_in_rectangle(ref, params)
        if rule == "object_near_point":
            return self._object_near_point(ref, params)
        if rule == "object_in_circle":
            return self._object_in_circle(ref, params)
        if rule == "object_near_initial_position":
            actor_position = self._actor_position(self._resolve_actor(ref))
            target = self._episode_success_targets.get("house5_initial_position")
            if actor_position is None or target is None:
                return False
            dimensions = 2 if params.get("metric", "xyz") == "xy" else 3
            return bool(np.linalg.norm(actor_position[:dimensions] - target[:dimensions]) <= float(params.get("threshold", 0.05)))
        if rule == "lid_open":
            return self._lid_is_open(ref)
        if rule == "plug_inserted":
            return self._plug_matches_socket(self._resolve_plug_actor(ref), self._resolve_socket_actor(params.get("socket_ref")), params)
        return False

    def _object_touched(self, actor: Any | None, *, min_force: float = 0.05) -> bool:
        """Return whether any gripper finger is in contact with ``actor``.

        This is intentionally weaker than ``object_grasped``: one finger is
        enough and no opposing-force/angle test is applied.  It therefore
        credits the approach/contact stage without treating a touch as a
        successful grasp or as object motion.
        """
        if actor is None or self.raw_env is None:
            return False
        agent = getattr(self.raw_env, "agent", None)
        agents = getattr(agent, "agents", None)
        if agents is None:
            agents = [agent] if agent is not None else []
        targets = [actor]
        # Boxes/lids in the industrial task are articulations rather than
        # plain Actors.  Query every link so touching the lid or its body is
        # credited consistently with touching a free rigid object.
        get_links = getattr(actor, "get_links", None)
        if callable(get_links):
            try:
                targets.extend(list(get_links()))
            except Exception:
                pass
        checked = False
        for candidate in agents:
            scene = getattr(candidate, "scene", None)
            if scene is None:
                continue
            for link_name in ("finger1_link", "finger2_link", "finger1_tip", "finger2_tip"):
                link = getattr(candidate, link_name, None)
                if link is None:
                    continue
                for target in targets:
                    checked = True
                    try:
                        forces = _to_numpy(scene.get_pairwise_contact_forces(link, target))
                        if forces.size >= 3:
                            vectors = forces.reshape(-1, 3)
                            if bool(np.any(np.linalg.norm(vectors, axis=1) >= min_force)):
                                return True
                    except Exception:
                        continue
        return False

    def _update_progress(self) -> None:
        newly_achieved: list[str] = []
        achieved_ids = {item["id"] for item in self._progress_milestones if item["achieved"]}
        for item in self._progress_milestones:
            if item["achieved"] or any(dep not in achieved_ids for dep in item.get("requires", [])):
                continue
            milestone_id = item["id"]
            observed = self._progress_rule_observed(item)
            self._progress_stable_counts[milestone_id] = self._progress_stable_counts[milestone_id] + 1 if observed else 0
            if self._progress_stable_counts[milestone_id] >= int(item.get("params", {}).get("stable_steps", 1)):
                item["achieved"] = True
                item["first_step"] = self._progress_step
                newly_achieved.append(milestone_id)
                achieved_ids.add(milestone_id)
        self._progress_completion = sum(float(item["weight"]) for item in self._progress_milestones if item["achieved"])
        if newly_achieved:
            self._progress_trace.append({"step": self._progress_step, "completion": round(self._progress_completion, 6), "new_milestones": newly_achieved})

    def _force_progress_success(self) -> None:
        newly_achieved = []
        for item in self._progress_milestones:
            if not item["achieved"]:
                item["achieved"] = True
                item["first_step"] = self._progress_step
                item["forced_by_success"] = True
                newly_achieved.append(item["id"])
        self._progress_completion = 1.0
        if newly_achieved:
            self._progress_trace.append({"step": self._progress_step, "completion": 1.0, "new_milestones": newly_achieved, "forced_by_success": True})

    def _progress_info(self) -> dict[str, Any]:
        return {
            "version": self._progress_config.get("version"),
            "aggregation": self._progress_config.get("aggregation"),
            "process_completion": round(float(self._progress_completion), 6),
            "milestones": [
                {"id": item["id"], "weight": item["weight"], "achieved": bool(item["achieved"]), "first_step": item["first_step"], **({"forced_by_success": True} if item.get("forced_by_success") else {})}
                for item in self._progress_milestones
            ],
            "trace": list(self._progress_trace),
        }

    def _reset_options(self, cfg: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.spec.key != "bio2":
            return None
        layout = cfg.get("layout_config", {})
        return {
            "layout_config": layout,
            "target_container": cfg.get("target_container"),
            "red_pad_pos": layout.get("red_pad_pos"),
            "yellow_pad_pos": layout.get("yellow_pad_pos"),
        }

    def _prepare_reset(self, cfg: Mapping[str, Any]) -> None:
        if self.raw_env is None:
            return
        if self.spec.key == "bio4":
            self.raw_env.beaker_position_index = cfg.get("beaker_position_index")

    def _apply_episode_config(self, cfg: Mapping[str, Any]) -> None:
        if self.raw_env is None:
            return
        if self.spec.key == "bio4":
            self._apply_bio4(cfg)
        elif self.spec.key == "bio5":
            self._apply_bio5(cfg)
        elif self.spec.key == "ind2":
            self._apply_ind2(cfg)
        elif self.spec.key == "ind3":
            self._apply_ind3(cfg)
        elif self.spec.key == "ind5":
            self._apply_ind5(cfg)
        elif self.spec.key == "house1":
            self._apply_household_layout("plate", cfg)
            self._apply_house1_cover_state()

    def _requires_rebuild(self, cfg: Mapping[str, Any]) -> bool:
        if self.env is None:
            return False
        if self.spec.key == "house5":
            return int(cfg.get("layout_id", 0)) != int(
                getattr(self.raw_env, "layout_id", self.spec.default_episode.get("layout_id", 0))
            )
        if self.spec.key == "ind5":
            layout = cfg.get("layout_config", {})
            expected_types = list(layout.get("part_types", {}).get(part, "cube") for part in layout.get("parts", []))
            return (
                int(getattr(self.raw_env, "num_parts", 0)) != len(expected_types)
                or list(getattr(self.raw_env, "part_types", [])) != expected_types
            )
        return False

    def _apply_bio4(self, cfg: Mapping[str, Any]) -> None:
        # The source scene consumes this value during reset.  It is assigned
        # before reset in _prepare_reset and repeated here for visibility.
        self.raw_env.beaker_position_index = cfg.get("beaker_position_index")

    def _apply_bio5(self, cfg: Mapping[str, Any]) -> None:
        from mani_skill.envs.tasks.custom_task.biolab.Scene_L5_02 import (
            LAYOUT_8,
            cube_origin_place,
        )
        from mani_skill.utils.structs.pose import Pose as MPose

        layout_id = int(cfg.get("layout_id", 0)) % len(LAYOUT_8)
        slots = LAYOUT_8[layout_id]
        for tube_index, slot_index in enumerate(slots):
            actor = self._object_actor(f"sample_tube_{tube_index}")
            if actor is None:
                continue
            actor.set_pose(
                MPose.create_from_pq(
                    torch.tensor(
                        [cube_origin_place[slot_index]],
                        dtype=torch.float32,
                        device=self.raw_env.device,
                    )
                )
            )

    def _apply_household_layout(self, prefix: str, cfg: Mapping[str, Any]) -> None:
        if self.spec.key == "house1":
            from mani_skill.envs.tasks.custom_task.household.scene_H1_01 import (
                SceneH1_01Env as EnvClass,
            )
        else:
            from mani_skill.envs.tasks.custom_task.household.scene_H3_04 import (
                SceneH3_04Env as EnvClass,
            )
        from mani_skill.utils.structs.pose import Pose as MPose

        layouts = (
            EnvClass.PREDEFINED_PLATE_POSITIONS
            if prefix == "plate"
            else EnvClass.PREDEFINED_BOWL_POSITIONS
        )
        layout_id = int(cfg.get("layout_id", 0)) % len(layouts)
        positions = layouts[layout_id]
        for index, row in enumerate(positions):
            actor = self._indexed_actor(prefix, index)
            if actor is None:
                continue
            pos = row["pos"] if isinstance(row, Mapping) else row
            quaternion = self._source_layout_quaternion(row)
            actor.set_pose(
                MPose.create_from_pq(
                    torch.tensor([pos], dtype=torch.float32, device=self.raw_env.device),
                    torch.tensor(
                        [quaternion], dtype=torch.float32, device=self.raw_env.device
                    ),
                )
            )
        # H1's source collector rebuilds the scene with only the plates in a
        # selected layout. This environment always owns three plate actors, so
        # remove any source-absent plate from simulation and the camera.
        if self.spec.key == "house1" and prefix == "plate":
            for index in range(len(positions), 3):
                actor = self._indexed_actor(prefix, index)
                if actor is None:
                    continue
                actor.set_pose(
                    MPose.create_from_pq(
                        torch.tensor(
                            [[10.0, 10.0, 10.0]],
                            dtype=torch.float32,
                            device=self.raw_env.device,
                        )
                    )
                )

    @staticmethod
    def _source_layout_quaternion(row: Any) -> np.ndarray:
        """Match the collectors' raw scipy quaternion passed to sapien.Pose."""
        if not isinstance(row, Mapping) or "rot_deg" not in row:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        roll, pitch, yaw = np.deg2rad(np.asarray(row["rot_deg"], dtype=np.float64))
        cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
        cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        # The source scripts pass scipy's [x, y, z, w] array directly to
        # sapien.Pose. Keep that exact convention here for trajectory parity.
        return np.asarray(
            [
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ],
            dtype=np.float32,
        )

    def _apply_house1_cover_state(self) -> None:
        """Apply the source script's reset-time baffle placement."""
        # ``baffle`` is a static actor, so ManiSkill reset does not restore its
        # pose after the demonstration's ``remove_occluder`` event moves it
        # off-table.  Reapply the complete source initial pose on every reset
        # before selecting the side for the current covered plate.
        baffle = getattr(self.raw_env, "baffle", None)
        if baffle is None:
            return
        history = self._public_episode.get("history_events", [])
        cover = next(
            (
                event
                for event in history
                if isinstance(event, Mapping) and event.get("type") == "cover"
            ),
            None,
        )
        position = self._actor_position(
            self._resolve_actor(cover.get("object"))
        ) if cover is not None else None
        side = 0.2 if position is not None and float(position[1]) > 0.0 else -0.2
        baffle.set_pose(
            sapien.Pose(
                p=np.asarray([-0.35, side, 0.15], dtype=np.float32),
                q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        )

    def _stabilize_custom_scene(self) -> None:
        """Run source collector setup motions that are not demonstration frames."""
        if self.spec.key in {"bio2", "ind2", "ind3"}:
            # The source collector closes both grippers for 40 control steps
            # before beginning its scene-only history.
            steps = 40
        elif self.spec.key in {"bio4", "house1", "house3"}:
            steps = 50
        else:
            return
        for _ in range(steps):
            self._custom_collection_step()

    def _custom_collection_step(self) -> None:
        """Use the source collectors' held-arm, closed-gripper action."""
        agent = getattr(self.raw_env, "agent", None)
        agents = getattr(agent, "agents", None)
        if agents is None:
            return
        action = {
            uid: np.zeros(space.shape, dtype=np.float32)
            for uid, space in self.env.action_space.items()
        }
        for index, robot in enumerate(agents):
            qpos = _to_numpy(robot.robot.get_qpos()).reshape(-1).copy()
            if qpos.size >= 7:
                qpos[6] = -1.0
            uid = f"piper_x-{index}"
            if uid in action:
                action[uid] = qpos[: len(action[uid])]
        self.env.step(action)

    def _reset_wrapper_elapsed_steps(self) -> None:
        """Keep source setup/idle frames outside the benchmark action budget."""
        wrapper = self.env
        seen: set[int] = set()
        while wrapper is not None and id(wrapper) not in seen:
            seen.add(id(wrapper))
            if hasattr(wrapper, "_elapsed_steps"):
                # Gym wrappers commonly use a Python integer, while ManiSkill's
                # BaseEnv uses a per-environment torch tensor.  Preserve the
                # counter's type/shape so the next BaseEnv.reset() can still
                # index it with env_idx.
                elapsed_steps = wrapper._elapsed_steps
                if isinstance(elapsed_steps, torch.Tensor):
                    elapsed_steps.zero_()
                elif isinstance(elapsed_steps, np.ndarray):
                    elapsed_steps[...] = 0
                else:
                    wrapper._elapsed_steps = 0
            wrapper = getattr(wrapper, "env", None)

    def _apply_ind2(self, cfg: Mapping[str, Any]) -> None:
        env = self.raw_env
        layout = cfg.get("layout_config", {})
        box_positions = layout.get("box_positions", {})
        env.num_boxes = int(layout.get("num_boxes", env.num_boxes))
        for i in range(env.num_boxes):
            label = f"B{i + 1}"
            pos = box_positions.get(label)
            if pos is not None:
                env.box[i].set_pose(sapien.Pose(p=pos))
        for i in range(env.num_boxes, env.max_num_boxes):
            env.box[i].set_pose(sapien.Pose(p=[10, 10, 10], q=[1, 0, 0, 0]))

        skip_box = cfg.get("skip_box")
        env.skipped_idx = self._name_index(skip_box, "B") if skip_box else env.skipped_idx
        if env.skipped_idx is None or env.skipped_idx < 0 or env.skipped_idx >= env.num_boxes:
            env.skipped_idx = 0
        env.skipped_box = env.box[env.skipped_idx]
        env.skipped_box_name = env.skipped_box.name

        open_order = []
        for name in cfg.get("open_boxes", []) or []:
            idx = self._name_index(name, "B")
            if idx is not None and idx < env.num_boxes:
                open_order.append(idx)
        env.open_order = open_order
        for box in env.box[: env.num_boxes]:
            env.close_lid(box)

    def _apply_ind3(self, cfg: Mapping[str, Any]) -> None:
        env = self.raw_env
        layout = cfg.get("layout_config", {})
        socket_list = layout.get("sockets", [])
        socket_positions = layout.get("positions", {})
        env.num_sockets = len(socket_list)
        loose_set = set(cfg.get("loose_peg", []) or [])
        remove_idx = self._name_index(cfg.get("remove_from"), "S")
        target_socket = cfg.get(
            "target_location_ids", self._evaluator_config.get("socket_ref")
        )
        target_idx = self._name_index(target_socket, "S")
        if target_idx is None:
            target_idx = remove_idx if remove_idx is not None else 0

        from mani_skill.utils.structs.pose import Pose as MPose

        socket_q = torch.tensor([[0.7071068, 0, 0.7071068, 0]], device=env.device)
        hide_p = torch.tensor([[10.0, 10.0, 10.0]], device=env.device)
        hide_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
        for i, socket_name in enumerate(socket_list):
            pos = socket_positions.get(socket_name)
            if pos is None:
                continue
            box_p = torch.tensor([pos], dtype=torch.float32, device=env.device)
            env.boxes[i].set_pose(MPose.create_from_pq(box_p, socket_q))

            rack_p = torch.tensor([[pos[0] + 0.2, pos[1], env._rack_outer_hs[2]]], device=env.device)
            env.plug_racks[i].set_pose(MPose.create_from_pq(rack_p))

        # Match collect_plug_restoration.py: Sx owns env.pegs[x - 1].
        # The table's PegN is a query label and does not remap scene actors.
        for peg_index, socket_name in enumerate(socket_list):
            if socket_name == target_socket and cfg.get("remove_from") != socket_name:
                env.pegs[peg_index].set_pose(MPose.create_from_pq(hide_p, hide_q))
                continue
            pos = socket_positions.get(socket_name)
            if pos is None:
                env.pegs[peg_index].set_pose(MPose.create_from_pq(hide_p, hide_q))
                continue
            if socket_name in loose_set:
                peg_p = torch.tensor(
                    [[pos[0] + 0.2, pos[1], env._rack_outer_hs[2] * 2]],
                    dtype=torch.float32,
                    device=env.device,
                )
            else:
                peg_p = torch.tensor(
                    [[pos[0], pos[1], 0.022]],
                    dtype=torch.float32,
                    device=env.device,
                )
            env.pegs[peg_index].set_pose(MPose.create_from_pq(peg_p, socket_q))

        for i in range(env.num_sockets, len(env.boxes)):
            pose = MPose.create_from_pq(hide_p, hide_q)
            env.boxes[i].set_pose(pose)
            env.plug_racks[i].set_pose(pose)
            env.pegs[i].set_pose(pose)

        env.removed_plug_idx = target_idx
        if remove_idx is not None and remove_idx != target_idx:
            env.removed_plug = env.pegs[remove_idx]
        else:
            env.removed_plug = env.pegs[target_idx]
        env.removed_plug_name = env.removed_plug.name

    def _apply_ind5(self, cfg: Mapping[str, Any]) -> None:
        env = self.raw_env
        layout = cfg.get("layout_config", {})
        parts = layout.get("parts", [])
        positions = layout.get("positions", {})
        part_to_cube = {part: env.cubes[i] for i, part in enumerate(parts[: len(env.cubes)])}
        from mani_skill.utils.structs.pose import Pose as MPose

        for part, cube in part_to_cube.items():
            pos = positions.get(part)
            if pos is not None:
                cube.set_pose(MPose.create_from_pq(torch.tensor([pos], dtype=torch.float32, device=env.device)))

    def _fresh_obs(self, fallback: Any) -> Any:
        try:
            return self.raw_env.get_obs()
        except Exception:
            return fallback

    def _replay_demonstration(self) -> dict[str, Any]:
        replayer = SceneDemonstrationReplayer(
            task_key=self.spec.key,
            raw_env=self.raw_env,
            cfg=self.episode_config,
            events=self._public_episode.get("history_events", []),
            resolve_actor=self._resolve_actor,
            on_frame=self._demonstration_frame_callback,
            source_camera="human_cam" if self.spec.key in {"bio4", "house1", "house3"} else None,
            initial_frame_step=self._custom_collection_step
            if self.spec.key in {"bio4", "house1", "house3"}
            else None,
        )
        try:
            demonstration = replayer.replay()
        except Exception as exc:
            raise DemonstrationReplayError(
                f"Could not replay scene demonstration for {self.spec.key} "
                f"episode {self._public_episode.get('episode_id')}: {exc}"
            ) from exc
        self.last_demonstration = demonstration
        return demonstration

    @staticmethod
    def _assert_interaction_frame_matches_observation(
        demonstration: Mapping[str, Any], obs: Any
    ) -> None:
        if not isinstance(obs, Mapping):
            return
        frames = demonstration.get("frames")
        camera = demonstration.get("camera")
        sensor_data = obs.get("sensor_data")
        if (
            not isinstance(frames, np.ndarray)
            or not frames.size
            or not isinstance(camera, str)
            or not isinstance(sensor_data, Mapping)
        ):
            return
        camera_data = sensor_data.get(camera)
        if not isinstance(camera_data, Mapping):
            return
        rgb = next(
            (
                camera_data[key]
                for key in ("rgb", "Color", "color")
                if key in camera_data
            ),
            None,
        )
        if rgb is None:
            return
        image = _to_numpy(rgb)
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            return
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = (
                    np.clip(image, 0.0, 1.0) * 255.0
                    if image.max(initial=0) <= 1.0
                    else np.clip(image, 0.0, 255.0)
                )
            image = image.astype(np.uint8)
        if not np.array_equal(frames[-1], image):
            raise RuntimeError(
                "Demonstration interaction frame does not match the post-demo observation."
            )

    def _capture_episode_success_targets(self) -> None:
        self._episode_success_targets = {}
        if self._evaluator_config.get("rule") != "object_near_initial_position":
            return
        actor = self._resolve_actor(self._evaluator_config.get("object_ref"))
        position = self._actor_position(actor)
        if position is not None:
            self._episode_success_targets["house5_initial_position"] = position

    def _capture_failure_baselines(self) -> None:
        """Record private interaction-start positions after demonstration replay."""
        self._failure_baseline_positions = {}
        self._failure_baseline_states = {}
        for rule_index, failure_rule in enumerate(
            self._evaluator_config.get("failure_rules", [])
        ):
            rule = failure_rule.get("rule")
            if rule in {
                "wrong_object_interaction",
                "wrong_plug_interaction",
            }:
                continue
            if rule == "wrong_lid_open":
                for object_ref in failure_rule["object_refs"]:
                    self._failure_baseline_states[(rule_index, object_ref)] = (
                        self._lid_is_open(object_ref)
                    )
            elif rule == "wrong_object_in_rectangle":
                for object_ref in failure_rule["object_refs"]:
                    self._failure_baseline_states[(rule_index, object_ref)] = (
                        self._object_in_rectangle(object_ref, failure_rule)
                    )
            elif rule == "wrong_object_near_point":
                for object_ref in failure_rule["object_refs"]:
                    self._failure_baseline_states[(rule_index, object_ref)] = (
                        self._object_near_point(object_ref, failure_rule)
                    )
            elif rule == "wrong_object_in_circle":
                for object_ref in failure_rule["object_refs"]:
                    self._failure_baseline_states[(rule_index, object_ref)] = (
                        self._object_in_circle(object_ref, failure_rule)
                    )
            elif rule == "wrong_plug_in_socket":
                for object_ref in failure_rule["object_refs"]:
                    plug = self._resolve_plug_actor(object_ref)
                    socket = self._resolve_socket_actor(failure_rule["socket_ref"])
                    self._failure_baseline_states[(rule_index, object_ref)] = (
                        self._plug_matches_socket(plug, socket, failure_rule)
                    )
            else:
                raise ValueError(f"Unsupported benchmark failure rule {rule!r}")

        for rule_index, failure_rule in enumerate(
            self._evaluator_config.get("failure_rules", [])
        ):
            if failure_rule.get("rule") not in {
                "wrong_object_interaction",
                "wrong_plug_interaction",
            }:
                continue
            for object_ref in failure_rule["object_refs"]:
                position = self._actor_position(self._resolve_actor(object_ref))
                if position is not None:
                    self._failure_baseline_positions[object_ref] = position.copy()
                self._failure_baseline_states[(rule_index, f"grasp:{object_ref}")] = (
                    self._grasp_state(self._resolve_actor(object_ref)) is True
                )

    def _benchmark_failure(self) -> bool:
        """Return whether a private immediate-failure rule is currently met."""
        if self._failure_latched:
            return True
        for rule_index, failure_rule in enumerate(
            self._evaluator_config.get("failure_rules", [])
        ):
            rule = failure_rule.get("rule")
            triggered_ref = None
            if rule in {"wrong_object_interaction", "wrong_plug_interaction"}:
                triggered_ref = self._wrong_object_interaction(
                    rule_index, failure_rule
                )
            elif rule == "wrong_object_in_rectangle":
                triggered_ref = self._first_newly_violating_object(
                    rule_index, failure_rule, self._object_in_rectangle
                )
            elif rule == "wrong_object_near_point":
                triggered_ref = self._first_newly_violating_object(
                    rule_index, failure_rule, self._object_near_point
                )
            elif rule == "wrong_object_in_circle":
                triggered_ref = self._first_newly_violating_object(
                    rule_index, failure_rule, self._object_in_circle
                )
            elif rule == "wrong_lid_open":
                triggered_ref = self._first_newly_violating_object(
                    rule_index,
                    failure_rule,
                    lambda object_ref, _rule: self._lid_is_open(object_ref),
                )
            elif rule == "wrong_plug_in_socket":
                triggered_ref = self._wrong_plug_in_socket(
                    rule_index, failure_rule
                )
            else:
                raise ValueError(f"Unsupported benchmark failure rule {rule!r}")
            if triggered_ref is not None:
                self._set_failure_reason(
                    rule, rule_index, failure_rule, triggered_ref
                )
                return True
        return False

    def _set_failure_reason(
        self,
        rule: str,
        rule_index: int,
        failure_rule: Mapping[str, Any],
        object_ref: str,
    ) -> None:
        """Keep evaluator diagnostics private while making failures auditable."""
        self._failure_reason = {
            "rule": rule,
            "rule_index": int(rule_index),
            "object_ref": object_ref,
        }
        if "socket_ref" in failure_rule:
            self._failure_reason["socket_ref"] = failure_rule["socket_ref"]

    def _wrong_object_interaction(
        self, rule_index: int, rule: Mapping[str, Any]
    ) -> str | None:
        lift_threshold = float(rule["lift_threshold"])
        move_threshold = float(rule["move_threshold"])
        for object_ref in rule["object_refs"]:
            actor = self._resolve_actor(object_ref)
            if actor is None:
                continue
            grasped = self._grasp_state(actor)
            if grasped is not None:
                if self._newly_violates(
                    rule_index=rule_index,
                    object_ref=f"grasp:{object_ref}",
                    current=grasped,
                ):
                    return object_ref
                # A reliable ``False`` grasp result does not mean the actor
                # was not moved. Position/lift checks run independently so
                # pushing, collisions, and dragging are still failures.
            initial = self._failure_baseline_positions.get(object_ref)
            current = self._actor_position(actor)
            if initial is None or current is None:
                continue
            if float(current[2] - initial[2]) > lift_threshold:
                return object_ref
            if float(np.linalg.norm(current - initial)) > move_threshold:
                return object_ref
        return None

    def _object_in_rectangle(self, object_ref: str, rule: Mapping[str, Any]) -> bool:
        center = _to_numpy(rule["center"]).reshape(-1)
        half_extents = _to_numpy(rule["half_extents"]).reshape(-1)
        dimensions = 2 if rule.get("metric", "xy") == "xy" else 3
        if center.size < dimensions or half_extents.size < dimensions:
            return False
        position = self._actor_position(self._resolve_actor(object_ref))
        return bool(
            position is not None
            and np.all(
                np.abs(position[:dimensions] - center[:dimensions])
                <= half_extents[:dimensions]
            )
        )

    def _object_near_point(self, object_ref: str, rule: Mapping[str, Any]) -> bool:
        target = _to_numpy(rule["target"]).reshape(-1)
        dimensions = 2 if rule.get("metric", "xyz") == "xy" else 3
        if target.size < dimensions:
            return False
        threshold = float(rule["threshold"])
        position = self._actor_position(self._resolve_actor(object_ref))
        return bool(
            position is not None
            and float(np.linalg.norm(position[:dimensions] - target[:dimensions]))
            <= threshold
        )

    def _object_in_circle(self, object_ref: str, rule: Mapping[str, Any]) -> bool:
        center = _to_numpy(rule["center"]).reshape(-1)
        dimensions = 2 if rule.get("metric", "xy") == "xy" else 3
        if center.size < dimensions:
            return False
        radius = float(rule["radius"])
        position = self._actor_position(self._resolve_actor(object_ref))
        return bool(
            position is not None
            and float(np.linalg.norm(position[:dimensions] - center[:dimensions]))
            <= radius
        )

    def _first_newly_violating_object(
        self,
        rule_index: int,
        rule: Mapping[str, Any],
        predicate: Callable[[str, Mapping[str, Any]], bool],
    ) -> str | None:
        for object_ref in rule["object_refs"]:
            if self._newly_violates(
                rule_index, object_ref, predicate(object_ref, rule)
            ):
                return object_ref
        return None

    def _lid_is_open(self, object_ref: str) -> bool:
        env = self.raw_env
        boxes = getattr(env, "box", [])
        index = self._name_index(object_ref, "B")
        return bool(
            index is not None
            and index >= 0
            and index < len(boxes)
            and env.is_lid_open(boxes[index])
        )

    def _wrong_plug_in_socket(
        self, rule_index: int, rule: Mapping[str, Any]
    ) -> str | None:
        socket = self._resolve_socket_actor(rule["socket_ref"])
        if socket is None:
            return None
        for object_ref in rule["object_refs"]:
            plug = self._resolve_plug_actor(object_ref)
            if self._newly_violates(
                rule_index,
                object_ref,
                self._plug_matches_socket(plug, socket, rule),
            ):
                return object_ref
        return None

    def _newly_violates(
        self, rule_index: int, object_ref: str, current: bool
    ) -> bool:
        key = (rule_index, object_ref)
        previous = self._failure_baseline_states.get(key, False)
        self._failure_baseline_states[key] = current
        return current and not previous

    def _grasp_state(self, actor: Any | None) -> bool | None:
        if actor is None:
            return None
        agent = getattr(self.raw_env, "agent", None)
        agents = getattr(agent, "agents", None)
        if agents is None:
            agents = [agent] if agent is not None else []
        checked = False
        for candidate in agents:
            grasp = getattr(candidate, "is_grasping", None)
            if not callable(grasp):
                continue
            try:
                grasped = _to_bool(grasp(actor))
                checked = True
                if grasped:
                    return True
            except Exception:
                continue
        return False if checked else None

    def _benchmark_success(self) -> bool:
        """Return the public benchmark outcome according to ``success_mode``.

        ``ever`` (the default) latches the first successful state so running
        with ``stop_on_success=false`` cannot turn a successful episode back
        into a failure. ``final`` preserves final-state semantics.
        """
        current = self._benchmark_success_current()
        if self._failure_latched:
            return False
        if current:
            self._success_latched = True
        return self._success_latched if self.success_mode == "ever" else current

    def _benchmark_success_current(self) -> bool:
        rule = self._evaluator_config.get("rule", self.spec.evaluator_rule)
        if rule == "objects_in_pad":
            return self._success_bio2()
        if rule == "object_in_rectangle":
            return self._success_bio4()
        if rule == "object_near_point":
            return self._success_object_near_point()
        if rule == "lid_open":
            return self._success_ind2()
        if rule == "object_in_circle":
            return self._success_ind5()
        if rule == "object_near_initial_position":
            return self._success_house5()
        if rule == "plug_inserted":
            return self._success_plug_inserted()
        raise ValueError(f"Unsupported benchmark evaluator rule {rule!r}")

    def _success_object_near_point(self) -> bool:
        actor = self._resolve_actor(self._evaluator_config.get("object_ref"))
        return self._within_target(actor, self._evaluator_config.get("target"))

    def _success_bio2(self) -> bool:
        target_objects = self._evaluator_config.get("object_refs", [])
        center = self._evaluator_config.get("center")
        half_extents = self._evaluator_config.get("half_extents")
        if isinstance(target_objects, str):
            target_objects = [target_objects]
        if not target_objects or center is None or half_extents is None:
            return False
        for name in target_objects:
            if not self._object_in_rectangle(name, self._evaluator_config):
                return False
        return True

    def _success_bio4(self) -> bool:
        actor = self._resolve_actor(self._evaluator_config.get("object_ref"))
        position = self._actor_position(actor)
        tray_center = self._evaluator_config.get("center")
        tray_half_size = self._evaluator_config.get("half_extents")
        if position is None or tray_center is None or tray_half_size is None:
            return False
        center = _to_numpy(tray_center).reshape(-1)
        half_size = _to_numpy(tray_half_size).reshape(-1)
        if center.size < 2 or half_size.size < 2:
            return False
        dimensions = 2 if self._evaluator_config.get("metric", "xy") == "xy" else 3
        if center.size < dimensions or half_size.size < dimensions or position.size < dimensions:
            return False
        return bool(np.all(np.abs(position[:dimensions] - center[:dimensions]) <= half_size[:dimensions]))

    def _success_ind2(self) -> bool:
        env = self.raw_env
        idx = self._name_index(self._evaluator_config.get("object_ref"), "B")
        boxes = getattr(env, "box", []) if env is not None else []
        active_count = int(getattr(env, "num_boxes", len(boxes))) if env is not None else 0
        if idx is None or idx < 0 or idx >= len(boxes) or idx >= active_count:
            return False
        return bool(env.is_lid_open(boxes[idx]))

    def _success_plug_inserted(self) -> bool:
        """Evaluate ind3 solely from the manifest's geometric rule parameters."""
        plug = self._resolve_plug_actor(self._evaluator_config.get("object_ref"))
        socket = self._resolve_socket_actor(self._evaluator_config.get("socket_ref"))
        return self._plug_matches_socket(plug, socket, self._evaluator_config)

    @staticmethod
    def _plug_matches_socket(
        plug: Any | None, socket: Any | None, rule: Mapping[str, Any]
    ) -> bool:
        if plug is None or socket is None:
            return False
        try:
            plug_local_pose = socket.pose.sp.inv() * plug.pose.sp
            local_position = _to_numpy(plug_local_pose.p).reshape(-1)
        except Exception:
            return False
        if local_position.size < 3:
            return False
        lateral_distance = float(np.hypot(local_position[1], local_position[2]))
        insert_depth = -float(local_position[0])
        return bool(
            lateral_distance <= float(rule["lateral_threshold"])
            and insert_depth >= float(rule["min_insert_depth"])
        )

    def _success_ind5(self) -> bool:
        target_cube = self._resolve_actor(self._evaluator_config.get("object_ref"))
        if target_cube is None:
            return False
        pos = _to_numpy(target_cube.pose.sp.p).reshape(-1)
        center = self._evaluator_config.get("center")
        radius = self._evaluator_config.get("radius")
        if center is None or radius is None:
            return False
        center = _to_numpy(center).reshape(-1)
        radius = float(radius)
        dx = float(pos[0]) - float(center[0])
        dy = float(pos[1]) - float(center[1])
        dimensions = 2 if self._evaluator_config.get("metric", "xy") == "xy" else 3
        if pos.size < dimensions or center.size < dimensions:
            return False
        if dimensions == 3:
            dz = float(pos[2]) - float(center[2])
            return (dx * dx + dy * dy + dz * dz) ** 0.5 <= radius
        return (dx * dx + dy * dy) ** 0.5 <= radius

    def _success_house5(self) -> bool:
        actor = self._resolve_actor(self._evaluator_config.get("object_ref"))
        target = self._episode_success_targets.get("house5_initial_position")
        return self._within_target(actor, target)

    def _biolab_actor(self, name: Any) -> Any | None:
        actors = getattr(self.raw_env, "biolab_objects", None)
        return actors.get(name) if isinstance(actors, Mapping) else None

    def _object_actor(self, name: str) -> Any | None:
        objects = getattr(self.raw_env, "objects", None)
        if isinstance(objects, Mapping):
            return objects.get(name)
        return None

    def _resolve_actor(self, object_ref: Any) -> Any | None:
        if not isinstance(object_ref, str) or self.raw_env is None:
            return None

        direct_actor = getattr(self.raw_env, object_ref, None)
        if direct_actor is not None:
            return direct_actor

        actor = self._biolab_actor(object_ref)
        if actor is not None:
            return actor

        actor = self._object_actor(object_ref)
        if actor is not None:
            return actor

        if object_ref in {"wash_bottle", "source"}:
            return getattr(self.raw_env, "source", None)

        if object_ref.startswith("C"):
            index = self._name_index(object_ref, "C")
            beakers = getattr(self.raw_env, "beakers", [])
            if index is not None and index >= 0 and index < len(beakers):
                return beakers[index]
        if object_ref.startswith("B"):
            index = self._name_index(object_ref, "B")
            boxes = getattr(self.raw_env, "box", [])
            if index is not None and index >= 0 and index < len(boxes):
                return boxes[index]
        if object_ref.startswith("Peg"):
            return self._resolve_plug_actor(object_ref)
        if object_ref.startswith("S"):
            return self._resolve_plug_actor(object_ref)
        if object_ref.startswith("P"):
            layout = self.episode_config.get("layout_config", {})
            parts = list(layout.get("parts", []))
            cubes = getattr(self.raw_env, "cubes", [])
            try:
                index = parts.index(object_ref)
            except ValueError:
                return None
            return cubes[index] if index < len(cubes) else None
        return None

    def _resolve_plug_actor(self, object_ref: Any) -> Any | None:
        index = self._name_index(object_ref, "Peg")
        if index is None:
            index = self._name_index(object_ref, "S")
        pegs = getattr(self.raw_env, "pegs", [])
        return pegs[index] if index is not None and index >= 0 and index < len(pegs) else None

    def _resolve_socket_actor(self, object_ref: Any) -> Any | None:
        index = self._name_index(object_ref, "S")
        sockets = getattr(self.raw_env, "boxes", [])
        return sockets[index] if index is not None and index >= 0 and index < len(sockets) else None

    def _indexed_actor(self, prefix: str, index: Any) -> Any | None:
        try:
            actor_index = int(index) + 1
        except (TypeError, ValueError):
            return None
        return getattr(self.raw_env, f"{prefix}{actor_index}", None)

    def _within_target(self, actor: Any | None, target: Any) -> bool:
        position = self._actor_position(actor)
        if position is None or target is None:
            return False
        target_position = _to_numpy(target).reshape(-1)
        metric = self._evaluator_config.get("metric", "xyz")
        dimensions = 2 if metric == "xy" else 3
        if target_position.size < dimensions or position.size < dimensions:
            return False
        threshold = float(self._evaluator_config.get("threshold", 0.05))
        distance = float(np.linalg.norm(position[:dimensions] - target_position[:dimensions]))
        return distance <= threshold

    @staticmethod
    def _actor_position(actor: Any | None) -> np.ndarray | None:
        if actor is None:
            return None
        pose = getattr(actor, "pose", None)
        if pose is None:
            return None
        spatial_pose = getattr(pose, "sp", None)
        position = getattr(spatial_pose, "p", None) if spatial_pose is not None else None
        if position is None:
            position = getattr(pose, "p", None)
        if position is None:
            return None
        array = _to_numpy(position)
        if array.ndim > 1:
            array = array.reshape(-1, array.shape[-1])[0]
        array = array.reshape(-1)
        return array[:3] if array.size >= 3 else None

    @staticmethod
    def _name_index(name: Any, prefix: str) -> int | None:
        if not isinstance(name, str) or not name.startswith(prefix):
            return None
        try:
            return int(name[len(prefix) :]) - 1
        except ValueError:
            return None
