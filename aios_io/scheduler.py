"""Simplified Trifecta scheduler with priority support."""
import heapq
import itertools
from typing import List, Optional, Tuple

from .task import Task


class Scheduler:
    """Schedules tasks in three priority queues."""

    def __init__(self) -> None:
        self._counter = itertools.count()
        self.red_queue: List[Tuple[int, int, Task]] = []
        self.blue_queue: List[Tuple[int, int, Task]] = []
        self.yellow_queue: List[Tuple[int, int, Task]] = []

    def add_task(self, task: Task) -> None:
        item = (task.priority, next(self._counter), task)
        if task.phase == "R":
            heapq.heappush(self.red_queue, item)
        elif task.phase == "B":
            heapq.heappush(self.blue_queue, item)
        elif task.phase == "Y":
            heapq.heappush(self.yellow_queue, item)
        else:
            raise ValueError("Phase must be R, B, or Y")

    def run_next(self, phase: str) -> Optional[Task]:
        if phase == "R" and self.red_queue:
            return heapq.heappop(self.red_queue)[2]
        if phase == "B" and self.blue_queue:
            return heapq.heappop(self.blue_queue)[2]
        if phase == "Y" and self.yellow_queue:
            return heapq.heappop(self.yellow_queue)[2]
        return None
