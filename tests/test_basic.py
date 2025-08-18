import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aios_io.cluster import Cluster
from aios_io.node import Node
from aios_io.task import Task
from aios_io.scheduler import Scheduler
from aios_io.orchestrator import Orchestrator
from aios_io import seed
import pytest


def test_cluster_add_and_list():
    cluster = Cluster("c1")
    node = Node("n1", 2)
    cluster.add_node(node)
    assert "CPU=2" in cluster.list_nodes()[0]


def test_scheduler_runs_tasks():
    sched = Scheduler()
    task = Task("t1", "R", lambda: None)
    sched.add_task(task)
    assert sched.run_next("R") is task


def test_scheduler_respects_priority():
    sched = Scheduler()
    low = Task("low", "R", lambda: None, priority=10)
    high = Task("high", "R", lambda: None, priority=1)
    sched.add_task(low)
    sched.add_task(high)
    assert sched.run_next("R") is high


def test_node_digest_logging(tmp_path):
    digest_file = tmp_path / "log.txt"
    node = Node("n2", 1, digest_path=str(digest_file))
    task = Task("work", "R", lambda: None)
    node.assign_task(task)
    node.run_tasks()
    entries = node.digest.read()
    assert entries[0]["node"] == "n2"
    assert entries[0]["task"] == "work"


def test_node_heartbeat_and_usage():
    node = Node("n3", 1)
    node.heartbeat()
    assert node.is_alive()
    usage = node.resource_usage()
    assert "cpu" in usage and "memory" in usage

