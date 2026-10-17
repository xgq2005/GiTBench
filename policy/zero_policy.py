"""Example external policy for scripts/eval_policy.py."""

from gitbench.policy import ZeroPolicy


def make_policy(action_space):
    return ZeroPolicy(action_space)
