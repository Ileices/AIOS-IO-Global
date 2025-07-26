"""Cluster manager for grouping nodes."""
from __future__ import annotations

from typing import Dict, List

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
