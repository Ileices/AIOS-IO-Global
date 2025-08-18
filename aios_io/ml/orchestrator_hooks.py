"""Hooks for orchestrating ML tasks across a cluster of nodes."""

from __future__ import annotations

from ..cluster import Cluster
from ..scheduler import Scheduler

from .compression_service import encode_and_enqueue


async def trigger_ml_tasks(
    cluster: Cluster,
    scheduler: Scheduler,
    payload: bytes,
) -> None:
    """Encode ``payload`` and schedule RBY tasks on every node.

    The payload is first compressed and queued as a decoding task in the
    scheduler. Afterwards, an RBY processing task is generated for each node in
    ``cluster`` and added to the scheduler.
    """

    encode_and_enqueue(payload, scheduler, name="payload", phase="B")

    # Import here to leverage any fallback available in ``aios_io.ml``
    from . import create_task  # type: ignore

    for node in cluster.nodes.values():
        task = await create_task(node)
        scheduler.add_task(task)


__all__ = ["trigger_ml_tasks"]
