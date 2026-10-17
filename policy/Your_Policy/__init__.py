"""Small, ready-to-copy GITBench policy template."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from gitbench.policy import BasePolicy, zero_action


class YourPolicy(BasePolicy):
    """Replace ``predict_action`` with model inference.

    The checked-in implementation is runnable without a model and emits zero
    actions. A policy author normally only changes ``__init__`` to load a
    checkpoint and ``predict_action`` to return one environment action.
    """

    def __init__(self, action_space: spaces.Space):
        self.action_space = action_space
        self.task_key: str | None = None
        self.episode_id: int | None = None
        self.instruction = ""
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
        episode = (info or {}).get("episode", {})
        self.instruction = str(episode.get("instruction", "")) if isinstance(episode, dict) else ""
        demonstration = (info or {}).get("demonstration", {})
        frames = demonstration.get("frames") if isinstance(demonstration, dict) else None
        self.demonstration_frame_count = int(len(frames)) if frames is not None else None

    def predict_action(self, obs: Any, info: dict[str, Any] | None = None) -> Any:
        """Return one action matching ``self.action_space``."""
        del obs, info
        return zero_action(self.action_space)

    def act(self, obs: Any, info: dict[str, Any] | None = None) -> Any:
        return self.predict_action(obs, info)


def make_policy(action_space: spaces.Space) -> YourPolicy:
    return YourPolicy(action_space)
