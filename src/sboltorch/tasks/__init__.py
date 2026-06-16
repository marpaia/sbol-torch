"""Tasks: the training-objective plug point."""

from __future__ import annotations

from .base import Task, build_task
from .supervised import SupervisedTask

__all__ = ["Task", "build_task", "SupervisedTask"]
