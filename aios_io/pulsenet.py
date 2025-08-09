"""Asynchronous peer-to-peer messaging layer for AIOS IO."""

from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable, Dict, Tuple


Handler = Callable[[str], Awaitable[None]]


class PulseNet:
    """Tiny message bus supporting retries and broadcast."""

    def __init__(self, retries: int = 3) -> None:
        self.peers: Dict[str, Tuple[str, int]] = {}
        self.handlers: Dict[str, Handler] = {}
        self.retries = retries

    # ------------------------------------------------------------------
    # Peer management
    def register_peer(self, name: str, host: str, port: int) -> None:
        self.peers[name] = (host, port)

    def register_handler(self, event: str, handler: Handler) -> None:
        """Register an asynchronous handler for an event type."""

        self.handlers[event] = handler

    # ------------------------------------------------------------------
    # Networking primitives
    async def start_server(self, host: str, port: int) -> None:
        """Start a JSON based TCP server and dispatch to handlers."""

        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            data = await reader.read(65536)
            if data:
                try:
                    message = json.loads(data.decode())
                    event = message.get("event")
                    payload = message.get("data")
                    handler = self.handlers.get(event)
                    if handler:
                        await handler(payload)
                except json.JSONDecodeError:
                    pass
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(_handle, host, port)
        async with server:
            await server.serve_forever()

    async def _send_bytes(self, host: str, port: int, data: bytes) -> None:
        for attempt in range(self.retries):
            try:
                reader, writer = await asyncio.open_connection(host, port)
                writer.write(data)
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                break
            except OSError:
                if attempt == self.retries - 1:
                    raise
                await asyncio.sleep(0.1 * 2**attempt)

    async def send(self, name: str, event: str, data: str) -> None:
        """Send an event to a specific peer."""

        if name not in self.peers:
            raise ValueError("Unknown peer")
        host, port = self.peers[name]
        payload = json.dumps({"event": event, "data": data}).encode()
        await self._send_bytes(host, port, payload)

    async def broadcast(self, event: str, data: str) -> None:
        """Send an event to all known peers."""

        payload = json.dumps({"event": event, "data": data}).encode()
        await asyncio.gather(
            *(self._send_bytes(h, p, payload) for h, p in self.peers.values())
        )
