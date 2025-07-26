"""Simple command-line interface for AIOS IO."""
import argparse

from .cluster import Cluster
from .node import Node


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIOS IO Command Line")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add-node", help="Add a node to the cluster")
    add.add_argument("cluster")
    add.add_argument("node_id")
    add.add_argument("cpu", type=int)
    add.add_argument("--gpu", type=int, default=0)
    add.add_argument("--ram", type=int, default=0)

    list_nodes = sub.add_parser("list-nodes", help="List cluster nodes")
    list_nodes.add_argument("cluster")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cluster = Cluster(args.cluster)

    if args.command == "add-node":
        node = Node(args.node_id, args.cpu, args.gpu, args.ram)
        cluster.add_node(node)
        print(f"Added {node.info()} to cluster {cluster.name}")
    elif args.command == "list-nodes":
        print("\n".join(cluster.list_nodes()))


if __name__ == "__main__":
    main()
