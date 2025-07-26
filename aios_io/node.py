"""Basic node representation for AIOS IO."""
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Node:
    """Represents a compute node in the system."""
    node_id: str
    cpu_cores: int
    gpu_cores: int = 0
    ram_gb: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    def info(self) -> str:
        return (
            f"Node {self.node_id}: CPU={self.cpu_cores} cores, "
            f"GPU={self.gpu_cores} cores, RAM={self.ram_gb}GB"
        )
