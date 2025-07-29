"""Basic node representation for AIOS IO."""

from dataclasses import dataclass, field
from typing import Dict, List, Any

from .task import Task
from .digest import Digest


@dataclass
class Node:
    """Represents a compute node in the system."""

    node_id: str
    cpu_cores: int
    gpu_cores: int = 0
    ram_gb: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)
    tasks: List[Task] = field(default_factory=list)
    digest_path: str | None = None
    digest: Digest = field(init=False)

    def __post_init__(self) -> None:
        path = self.digest_path or f"digest_{self.node_id}.log"
        self.digest = Digest(path)

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
            self.digest.log(f"{self.node_id}:{task.name}")
        self.tasks.clear()

    # New functionality for persistence
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node information to a dictionary."""
        return {
            "node_id": self.node_id,
            "cpu_cores": self.cpu_cores,
            "gpu_cores": self.gpu_cores,
            "ram_gb": self.ram_gb,
            "metadata": self.metadata,
            "digest_path": str(self.digest.path),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Create a Node from a dictionary."""
        return cls(
            data["node_id"],
            data.get("cpu_cores", 0),
            data.get("gpu_cores", 0),
            data.get("ram_gb", 0),
            data.get("metadata", {}),
            data.get("digest_path"),
        )
