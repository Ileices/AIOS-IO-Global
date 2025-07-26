"""Basic placeholder for PulseNet communication layer."""
import asyncio
from typing import Dict, Tuple


class PulseNet:
    """Minimal peer registry using TCP for demonstration."""

    def __init__(self) -> None:
        self.peers: Dict[str, Tuple[str, int]] = {}

    def register_peer(self, name: str, host: str, port: int) -> None:
        self.peers[name] = (host, port)

    async def send(self, name: str, message: str) -> None:
        if name not in self.peers:
            raise ValueError("Unknown peer")
        host, port = self.peers[name]
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(message.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
