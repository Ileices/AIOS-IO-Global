"""Simple task representation for AIOS IO."""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    """Represents a unit of work with a phase and callable."""

    name: str
    phase: str
    action: Callable[[], None]

    def run(self) -> None:
        """Execute the task's action."""
        self.action()
