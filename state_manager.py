from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Lock, RLock
from typing import ClassVar
from uuid import uuid4

from algorithms.base import OptimizationAlgorithm


@dataclass(slots=True)
class ManagedTask:
    algorithm: OptimizationAlgorithm
    last_accessed: datetime


class TaskManager:
    """Thread-safe singleton for in-memory optimization task state."""

    _instance: ClassVar[TaskManager | None] = None
    _instance_lock: ClassVar[Lock] = Lock()

    def __new__(cls, *args: object, **kwargs: object) -> TaskManager:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, ttl: timedelta = timedelta(hours=1)) -> None:
        if getattr(self, "_is_initialized", False):
            return

        self._ttl = ttl
        self._tasks: dict[str, ManagedTask] = {}
        self._lock = RLock()
        self._is_initialized = True

    def create_task(self, algorithm: OptimizationAlgorithm) -> str:
        task_id = str(uuid4())
        now = datetime.now(UTC)
        with self._lock:
            self._tasks[task_id] = ManagedTask(algorithm=algorithm, last_accessed=now)
        return task_id

    def get_task(self, task_id: str) -> OptimizationAlgorithm:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"Task not found: {task_id}")
            task.last_accessed = datetime.now(UTC)
            return task.algorithm

    def remove_task(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)

    def cleanup_expired_tasks(self) -> list[str]:
        now = datetime.now(UTC)
        expired_task_ids: list[str] = []

        with self._lock:
            for task_id, task in list(self._tasks.items()):
                if now - task.last_accessed > self._ttl:
                    expired_task_ids.append(task_id)
                    del self._tasks[task_id]

        return expired_task_ids

    def active_task_count(self) -> int:
        with self._lock:
            return len(self._tasks)


_task_manager = TaskManager()


def get_task_manager() -> TaskManager:
    return _task_manager
