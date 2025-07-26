"""Simplified Trifecta scheduler."""
from collections import deque
from typing import Callable, Deque, Optional


class Scheduler:
    """Schedules tasks in three separate queues."""

    def __init__(self) -> None:
        self.red_queue: Deque[Callable] = deque()
        self.blue_queue: Deque[Callable] = deque()
        self.yellow_queue: Deque[Callable] = deque()

    def add_task(self, phase: str, task: Callable) -> None:
        if phase == "R":
            self.red_queue.append(task)
        elif phase == "B":
            self.blue_queue.append(task)
        elif phase == "Y":
            self.yellow_queue.append(task)
        else:
            raise ValueError("Phase must be R, B, or Y")

    def run_next(self, phase: str) -> Optional[any]:
        if phase == "R" and self.red_queue:
            return self.red_queue.popleft()()
        if phase == "B" and self.blue_queue:
            return self.blue_queue.popleft()()
        if phase == "Y" and self.yellow_queue:
            return self.yellow_queue.popleft()()
        return None
