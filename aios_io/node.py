"""Basic node representation for AIOS IO."""
from dataclasses import dataclass, field
from typing import Dict, List

from .task import Task

@dataclass
class Node:
    """Represents a compute node in the system."""
    node_id: str
    cpu_cores: int
    gpu_cores: int = 0
    ram_gb: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)
    tasks: List[Task] = field(default_factory=list)

    def info(self) -> str:
        return (
            f"Node {self.node_id}: CPU={self.cpu_cores} cores, "
            f"GPU={self.gpu_cores} cores, RAM={self.ram_gb}GB"
        )

    def assign_task(self, task: Task) -> None:
        """Assign a task to this node."""
        self.tasks.append(task)

    def run_tasks(self) -> None:
        """Execute and clear all assigned tasks."""
        for task in list(self.tasks):
            task.run()
        self.tasks.clear()
