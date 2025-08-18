"""Utility compression helpers for ML payloads.

This module provides a minimal interface for encoding data before it is fed
into the system's scheduler. Encoded payloads are wrapped in :class:`Task`
instances which are queued for later processing.
"""

from __future__ import annotations

import zlib
from typing import Optional

from ..scheduler import Scheduler
from ..task import Task


def encode_and_enqueue(
    payload: bytes,
    scheduler: Scheduler,
    *,
    name: str = "payload",
    phase: str = "B",
    priority: int = 0,
) -> bytes:
    """Compress ``payload`` and enqueue a decoding task.

    Parameters
    ----------
    payload:
        Raw data to be compressed.
    scheduler:
        Target :class:`~aios_io.scheduler.Scheduler` instance where a task
        capable of decoding the payload will be queued.
    name, phase, priority:
        Standard task parameters.

    Returns
    -------
    bytes
        The compressed payload.
    """

    encoded = zlib.compress(payload)

    def action() -> None:
        # The task's action simply decompresses the payload ensuring integrity.
        zlib.decompress(encoded)

    task = Task(name, phase, action, priority)
    scheduler.add_task(task)
    return encoded


__all__ = ["encode_and_enqueue"]
