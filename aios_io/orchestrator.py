"""High-level orchestrator that glues scheduler, cluster and PulseNet."""
from __future__ import annotations

import asyncio
from typing import Iterable, Optional

from .cluster import Cluster
from .scheduler import Scheduler
from .pulsenet import PulseNet


class Orchestrator:
    """Coordinate task scheduling across a cluster and optional network."""

    def __init__(self, cluster: Cluster, scheduler: Scheduler, pulsenet: Optional[PulseNet] = None) -> None:
        self.cluster = cluster
        self.scheduler = scheduler
        self.pulsenet = pulsenet

    def cycle(self, phases: Iterable[str] = ("R", "B", "Y")) -> None:
        """Run a full cycle of phases, dispatching tasks to nodes."""
        for phase in phases:
            task = self.scheduler.run_next(phase)
            if task:
                self.cluster.schedule_task(task)
            self.cluster.run_all()
            if self.pulsenet:
                asyncio.run(self.pulsenet.broadcast("phase-complete", phase))
