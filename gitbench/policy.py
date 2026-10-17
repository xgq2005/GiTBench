"""Minimal policy interface used by the benchmark scripts."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import numpy as np

# Gymnasium may be installed alongside (rather than inside) the RDT
# interpreter.  Add that package location only when it is not already
# importable; appending preserves the active environment's PyTorch precedence.
try:
    import gymnasium  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    extra_site = os.environ.get("MANISKILL_SITE_PACKAGES")
    if extra_site and os.path.isdir(extra_site) and extra_site not in sys.path:
        sys.path.append(extra_site)
from gymnasium import spaces


class BasePolicy:
    def reset(
        self,
        task: Any | None = None,
        episode_id: int | None = None,
        obs: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Receive reset context, including ``info["demonstration"]`` when present."""
        pass

    def act(self, obs: Any, info: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError


class ZeroPolicy(BasePolicy):
    def __init__(self, action_space: spaces.Space):
        self.action_space = action_space

    def act(self, obs: Any, info: dict[str, Any] | None = None) -> Any:
        return zero_action(self.action_space)


class RandomPolicy(BasePolicy):
    def __init__(self, action_space: spaces.Space):
        self.action_space = action_space

    def act(self, obs: Any, info: dict[str, Any] | None = None) -> Any:
        return self.action_space.sample()


def zero_action(space: spaces.Space) -> Any:
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, spaces.Dict):
        return {key: zero_action(subspace) for key, subspace in space.spaces.items()}
    if isinstance(space, spaces.Tuple):
        return tuple(zero_action(subspace) for subspace in space.spaces)
    raise TypeError(f"Do not know how to build a zero action for action space {space!r}")


def load_policy(spec: str, action_space: spaces.Space) -> BasePolicy:
    """Load a policy.

    Accepted values:
      - "zero"
      - "random"
      - "module:function", where function(action_space) returns an object with act().
      - "module:ClassName", where ClassName(action_space) creates the policy.
    """
    if spec == "zero":
        return ZeroPolicy(action_space)
    if spec == "random":
        return RandomPolicy(action_space)

    if ":" not in spec:
        raise ValueError("Policy must be 'zero', 'random', or 'module:factory'.")

    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    policy = factory(action_space)
    if not hasattr(policy, "act"):
        raise TypeError(f"Loaded policy from {spec} does not expose act(obs, info).")
    return policy
