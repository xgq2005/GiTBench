"""GITBench: lightweight benchmark adapters for the local ManiSkill sim tasks."""

from .tasks import TASKS, TaskSpec, get_task, list_tasks

__version__ = "0.4.0"

__all__ = ["TASKS", "TaskSpec", "get_task", "list_tasks", "__version__"]
