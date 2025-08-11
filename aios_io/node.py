"""Basic node representation for AIOS IO."""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import os
import time

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
    heartbeat_interval: float = 5.0
    last_heartbeat: float = field(init=False, default_factory=time.time)

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

            entry = {
                "node": self.node_id,
                "task": task.name,
                "timestamp": time.time(),
                "resource": self.resource_usage(),
            }
            self.digest.log(entry)
            self.heartbeat()

            self.digest.log(
                {"node": self.node_id, "task": task.name, "timestamp": time.time()}
            )

        self.tasks.clear()

    def cancel_task(self, name: str) -> bool:
        """Cancel a pending task by name."""
        for task in list(self.tasks):
            if task.name == name:
                self.tasks.remove(task)
                self.digest.log(
                    {
                        "node": self.node_id,
                        "task": name,
                        "timestamp": time.time(),
                        "status": "cancelled",
                    }
                )
                return True
        return False

    def heartbeat(self) -> None:
        """Record a heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def is_alive(self, timeout: float | None = None) -> bool:
        """Determine if the node is alive based on last heartbeat."""
        timeout = timeout or self.heartbeat_interval * 2
        return (time.time() - self.last_heartbeat) <= timeout

    def resource_usage(self) -> Dict[str, float]:
        """Estimate current CPU and memory usage as percentages."""
        load1, _, _ = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        cpu = (load1 / cpu_count) * 100
        meminfo = self._meminfo()
        mem_total = meminfo.get("MemTotal", 1)
        mem_available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        mem = 100.0 - (mem_available / mem_total * 100.0)
        return {"cpu": round(cpu, 2), "memory": round(mem, 2)}

    @staticmethod
    def _meminfo() -> Dict[str, int]:
        info: Dict[str, int] = {}
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    key, val = line.split(":", 1)
                    info[key] = int(val.strip().split()[0])
        except FileNotFoundError:
            pass
        return info

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
