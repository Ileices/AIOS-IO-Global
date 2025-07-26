"""Cluster manager for grouping nodes."""
from __future__ import annotations

from typing import Dict, List

from .node import Node


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
