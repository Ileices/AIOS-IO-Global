import asyncio
import contextlib
from pathlib import Path

from aios_io.pulsenet import PulseNet
from aios_io.cluster import Cluster
from aios_io.node import Node
from aios_io.scheduler import Scheduler
from aios_io.task import Task


async def _start_server(pn: PulseNet, host: str, port: int, store: list[str]) -> asyncio.Task:
    pn.register_handler("ping", lambda m: store.append(m))
    task = asyncio.create_task(pn.start_server(host, port))
    await asyncio.sleep(0.1)  # allow server to start
    return task


def test_network_and_digest(tmp_path: Path) -> None:
    async def run() -> None:
        # Start two PulseNet servers
        msgs1: list[str] = []
        pn1 = PulseNet()
        server1 = await _start_server(pn1, "127.0.0.1", 9101, msgs1)

        msgs2: list[str] = []
        pn2 = PulseNet()
        server2 = await _start_server(pn2, "127.0.0.1", 9102, msgs2)

        orchestrator = PulseNet()
        orchestrator.register_peer("n1", "127.0.0.1", 9101)
        orchestrator.register_peer("n2", "127.0.0.1", 9102)

        await orchestrator.broadcast("ping", "hello")
        await asyncio.sleep(0.1)

        assert msgs1 == ["hello"]
        assert msgs2 == ["hello"]

        # Stop servers
        server1.cancel()
        server2.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server1
        with contextlib.suppress(asyncio.CancelledError):
            await server2

        # Orchestrator cycle: schedule tasks across nodes
        cluster = Cluster("c")
        node1 = Node("n1", 2, digest_path=str(tmp_path / "n1.log"))
        node2 = Node("n2", 2, digest_path=str(tmp_path / "n2.log"))
        cluster.add_node(node1)
        cluster.add_node(node2)

        sched = Scheduler()
        sched.add_task(Task("collect", "R", lambda: None))
        sched.add_task(Task("train", "B", lambda: None))
        sched.add_task(Task("deploy", "Y", lambda: None))

        for phase in ["R", "B", "Y"]:
            task = sched.run_next(phase)
            if task:
                cluster.schedule_task(task)
        cluster.run_all()

        # Aggregate digests
        entries = node1.digest.read() + node2.digest.read()
        tasks_logged = {e["task"] for e in entries if "task" in e}
        assert {"collect", "train", "deploy"} <= tasks_logged

    asyncio.run(run())
