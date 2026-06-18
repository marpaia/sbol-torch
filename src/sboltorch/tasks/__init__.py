"""Tasks: the training-objective plug point."""

from __future__ import annotations

from sboltorch.tasks.base import Task, build_task
from sboltorch.tasks.supervised import SupervisedTask

__all__ = ["Task", "build_task", "SupervisedTask"]
