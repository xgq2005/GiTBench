"""Optional wrapper for integrations that manage two environments."""

from __future__ import annotations

from typing import Any

try:
    from .deploy_policy import YourModel, encode_obs, get_model, reset_model
except ImportError:  # Some launchers import deploy_policy as a top-level module.
    from deploy_policy import YourModel, encode_obs, get_model, reset_model


def eval(TASK_ENV: Any, model: YourModel, observation: Any) -> None:
    """Run one action chunk and update the model with each new observation."""
    obs = encode_obs(observation)
    if not model.obs_cache:
        model.update_obs(obs)
    for action in model.get_action():
        TASK_ENV.take_action(action)
        model.update_obs(encode_obs(TASK_ENV.get_obs()))


__all__ = ["YourModel", "encode_obs", "get_model", "reset_model", "eval"]
