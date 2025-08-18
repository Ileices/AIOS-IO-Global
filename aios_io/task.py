"""Simple task representation for AIOS IO."""
from dataclasses import dataclass, field
from typing import Callable, Any, Dict


@dataclass
class Task:
    """Represents a unit of work with a phase, callable, and priority."""

    name: str
    phase: str
    action: Callable[[], None]
    priority: int = 0
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def run(self) -> None:
        """Execute the task's action."""
        self.action()
