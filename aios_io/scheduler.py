"""Simplified Trifecta scheduler."""
from collections import deque
from typing import Deque, Optional

from .task import Task


class Scheduler:
    """Schedules tasks in three separate queues."""

    def __init__(self) -> None:
        self.red_queue: Deque[Task] = deque()
        self.blue_queue: Deque[Task] = deque()
        self.yellow_queue: Deque[Task] = deque()

    def add_task(self, task: Task) -> None:
        if task.phase == "R":
            self.red_queue.append(task)
        elif task.phase == "B":
            self.blue_queue.append(task)
        elif task.phase == "Y":
            self.yellow_queue.append(task)
        else:
            raise ValueError("Phase must be R, B, or Y")

    def run_next(self, phase: str) -> Optional[Task]:
        if phase == "R" and self.red_queue:
            return self.red_queue.popleft()
        if phase == "B" and self.blue_queue:
            return self.blue_queue.popleft()
        if phase == "Y" and self.yellow_queue:
            return self.yellow_queue.popleft()
        return None
