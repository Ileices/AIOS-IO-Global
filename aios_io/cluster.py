"""Cluster manager for grouping nodes."""
from __future__ import annotations

from typing import Dict, List, Any

from .node import Node
from .task import Task


class Cluster:
    """Manages a group of compute nodes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.nodes: Dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

    def list_nodes(self) -> List[str]:
        return [node.info() for node in self.nodes.values()]

    def schedule_task(self, task: Task) -> None:
        """Assign a task to the least-loaded node."""
        if not self.nodes:
            raise RuntimeError("Cluster has no nodes")
        # choose node with fewest tasks
        node = min(self.nodes.values(), key=lambda n: len(n.tasks))
        node.assign_task(task)

    def run_all(self) -> None:
        """Run all tasks on all nodes."""
        for node in self.nodes.values():
            node.run_tasks()

    # Persistence helpers
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the cluster to a dictionary."""
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cluster":
        """Create a cluster from serialized data."""
        cluster = cls(data["name"])
        for node_data in data.get("nodes", []):
            cluster.add_node(Node.from_dict(node_data))
        return cluster

    def save(self, path: str) -> None:
        """Save cluster configuration to JSON file."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Cluster":
        """Load cluster configuration from JSON file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
