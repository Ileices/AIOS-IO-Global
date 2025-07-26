"""Simple command-line interface for AIOS IO."""
import argparse

from .cluster import Cluster
from .node import Node
from .scheduler import Scheduler
from .task import Task


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

    demo = sub.add_parser("demo", help="Run a demo cycle")
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
    elif args.command == "demo":
        # create a small demo cluster
        cluster.add_node(Node("n1", 4))
        cluster.add_node(Node("n2", 2))
        scheduler = Scheduler()

        # demo tasks simply print their name
        scheduler.add_task(Task("collect", "R", lambda: print("collecting data")))
        scheduler.add_task(Task("train", "B", lambda: print("training model")))
        scheduler.add_task(Task("deploy", "Y", lambda: print("deploying service")))

        # schedule tasks onto nodes and run them
        for phase in ["R", "B", "Y"]:
            task = scheduler.run_next(phase)
            if task:
                cluster.schedule_task(task)
        cluster.run_all()


if __name__ == "__main__":
    main()
