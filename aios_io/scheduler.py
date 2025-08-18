"""Simplified Trifecta scheduler with priority support."""
from __future__ import annotations

import heapq
import itertools
from typing import Any, Dict, List, Optional, Tuple

from .task import Task
from .backends import BaseBackend, JSONFileBackend


class Scheduler:
    """Schedules tasks in three priority queues with persistence."""

    def __init__(self, backend: BaseBackend | None = None) -> None:
        self.backend = backend or JSONFileBackend()
        self._counter = itertools.count()
        self.red_queue: List[Tuple[float, int, Task]] = []
        self.blue_queue: List[Tuple[float, int, Task]] = []
        self.yellow_queue: List[Tuple[float, int, Task]] = []
        self.history: List[Dict[str, Any]] = []
        self._load_state()

    # ------------------------------------------------------------------
    def add_task(self, task: Task) -> None:
        """Add a task to the appropriate queue with weighted priority."""

        effective = task.priority / (task.weight or 1)
        item = (effective, next(self._counter), task)
        if task.phase == "R":
            heapq.heappush(self.red_queue, item)
        elif task.phase == "B":
            heapq.heappush(self.blue_queue, item)
        elif task.phase == "Y":
            heapq.heappush(self.yellow_queue, item)
        else:
            raise ValueError("Phase must be R, B, or Y")
        self._serialize_state()

    # ------------------------------------------------------------------
    def run_next(self, phase: str) -> Optional[Task]:
        """Run and record the next task for a phase."""

        queue_map = {"R": self.red_queue, "B": self.blue_queue, "Y": self.yellow_queue}
        queue = queue_map.get(phase)
        if not queue:
            return None
        task = heapq.heappop(queue)[2]
        record = self._task_to_dict(task)
        record["status"] = "ran"
        self.history.append(record)
        self._serialize_state()
        return task

    # ------------------------------------------------------------------
    def _task_to_dict(self, task: Task) -> Dict[str, Any]:
        return {
            "name": task.name,
            "phase": task.phase,
            "priority": task.priority,
            "weight": task.weight,
            "metadata": task.metadata,
        }

    def _serialize_state(self) -> None:
        data = {
            "queues": {
                "R": [self._task_to_dict(item[2]) for item in self.red_queue],
                "B": [self._task_to_dict(item[2]) for item in self.blue_queue],
                "Y": [self._task_to_dict(item[2]) for item in self.yellow_queue],
            },
            "history": self.history,
        }
        self.backend.save(data)

    def _load_state(self) -> None:
        data = self.backend.load()
        queues = data.get("queues", {})
        for phase, queue in [("R", self.red_queue), ("B", self.blue_queue), ("Y", self.yellow_queue)]:
            for meta in queues.get(phase, []):
                task = Task(
                    meta["name"],
                    meta["phase"],
                    lambda: None,
                    priority=meta.get("priority", 0),
                    weight=meta.get("weight", 1.0),
                    metadata=meta.get("metadata", {}),
                )
                effective = task.priority / (task.weight or 1)
                item = (effective, next(self._counter), task)
                heapq.heappush(queue, item)
        self.history = data.get("history", [])
