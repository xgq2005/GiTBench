"""Deterministic smoke-test policy for GITBench."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from gitbench.policy import BasePolicy, zero_action


class TestPolicy(BasePolicy):
    """A minimal policy that exercises the benchmark pipeline with zero actions."""

    def __init__(self, action_space: spaces.Space):
        self.action_space = action_space
        self.task_key: str | None = None
        self.episode_id: int | None = None
        self.demonstration_frame_count: int | None = None

    def reset(
        self,
        task: Any | None = None,
        episode_id: int | None = None,
        obs: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.task_key = getattr(task, "key", None)
        self.episode_id = episode_id
        demonstration = (info or {}).get("demonstration", {})
        frames = demonstration.get("frames") if isinstance(demonstration, dict) else None
        self.demonstration_frame_count = int(len(frames)) if frames is not None else None

    def act(self, obs: Any, info: dict[str, Any] | None = None) -> Any:
        return zero_action(self.action_space)


def make_policy(action_space: spaces.Space) -> TestPolicy:
    return TestPolicy(action_space)
