import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aios_io.cluster import Cluster
from aios_io.node import Node
from aios_io.task import Task
from aios_io.scheduler import Scheduler


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


def test_node_digest_logging(tmp_path):
    digest_file = tmp_path / "log.txt"
    node = Node("n2", 1, digest_path=str(digest_file))
    task = Task("work", "R", lambda: None)
    node.assign_task(task)
    node.run_tasks()
    assert digest_file.read_text().strip() == "n2:work"
