"""Simple command-line interface for AIOS IO."""

import argparse
import asyncio

from .cluster import Cluster
from .node import Node
from .scheduler import Scheduler
from .task import Task
from .pulsenet import PulseNet


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

    remove = sub.add_parser("remove-node", help="Remove a node from the cluster")
    remove.add_argument("cluster")
    remove.add_argument("node_id")

    save = sub.add_parser("save-cluster", help="Save cluster configuration")
    save.add_argument("cluster")
    save.add_argument("path")

    load = sub.add_parser("load-cluster", help="Load cluster configuration")
    load.add_argument("path")

    demo = sub.add_parser("demo", help="Run a demo cycle")
    demo.add_argument("cluster")

    reg = sub.add_parser("register-peer", help="Register a PulseNet peer")
    reg.add_argument("name")
    reg.add_argument("host")
    reg.add_argument("port", type=int)

    start = sub.add_parser("start-server", help="Start a PulseNet server")
    start.add_argument("host")
    start.add_argument("port", type=int)

    send = sub.add_parser("send", help="Send a message to a peer")
    send.add_argument("name")
    send.add_argument("message")

    digest_cmd = sub.add_parser("show-digest", help="Show node digest logs")
    digest_cmd.add_argument("cluster")

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    pulsenet = PulseNet()

    if args.command == "register-peer":
        pulsenet.register_peer(args.name, args.host, args.port)
        print(f"Registered peer {args.name} at {args.host}:{args.port}")
        return
    elif args.command == "start-server":
        asyncio.run(pulsenet.start_server(args.host, args.port, print))
        return
    elif args.command == "send":
        asyncio.run(pulsenet.send(args.name, args.message))
        return

    if args.command == "load-cluster":
        cluster = Cluster.load(args.path)
    else:
        cluster = Cluster(args.cluster)

    if args.command == "add-node":
        node = Node(args.node_id, args.cpu, args.gpu, args.ram)
        cluster.add_node(node)
        print(f"Added {node.info()} to cluster {cluster.name}")
    elif args.command == "list-nodes":
        print("\n".join(cluster.list_nodes()))
    elif args.command == "remove-node":
        cluster.remove_node(args.node_id)
        print(f"Removed node {args.node_id} from {cluster.name}")
    elif args.command == "save-cluster":
        cluster.save(args.path)
        print(f"Saved cluster {cluster.name} to {args.path}")
    elif args.command == "load-cluster":
        print(f"Loaded cluster {cluster.name}")
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
    elif args.command == "show-digest":
        for node in cluster.nodes.values():
            print(f"== {node.node_id} ==")
            for entry in node.digest.read():
                print(entry)


if __name__ == "__main__":
    main()
