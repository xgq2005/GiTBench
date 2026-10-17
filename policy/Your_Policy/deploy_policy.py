"""Optional model-hook template for GITBench deployments.

The normal entry point is ``__init__.py``. This module is for projects that
prefer a separate model object and an observation cache.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class YourModel:
    """Small deployment model with the same observation-cache contract."""

    def __init__(self, action_dim: int = 14, chunk_size: int = 1) -> None:
        self.action_dim = int(action_dim)
        self.chunk_size = max(1, int(chunk_size))
        self.obs_cache: list[Any] = []

    def update_obs(self, obs: Any) -> None:
        self.obs_cache.append(obs)

    def get_action(self) -> list[np.ndarray]:
        return [np.zeros(self.action_dim, dtype=np.float32) for _ in range(self.chunk_size)]


def encode_obs(observation: Any) -> Any:
    """Normalize an evaluator observation before passing it to the model."""
    return observation


def get_model(usr_args: Any) -> YourModel:
    """Build the placeholder model from deployment arguments."""
    action_dim = int(getattr(usr_args, "action_dim", 14))
    chunk_size = int(getattr(usr_args, "chunk_size", 1))
    return YourModel(action_dim=action_dim, chunk_size=chunk_size)


def eval(TASK_ENV: Any, model: YourModel, observation: Any) -> None:
    """Run one model action chunk against a compatible environment wrapper."""
    obs = encode_obs(observation)
    if not model.obs_cache:
        model.update_obs(obs)

    for action in model.get_action():
        TASK_ENV.take_action(action, action_type="qpos")
        obs = encode_obs(TASK_ENV.get_obs())
        model.update_obs(obs)


def reset_model(model: YourModel) -> None:
    """Clear per-episode observation state."""
    model.obs_cache.clear()
