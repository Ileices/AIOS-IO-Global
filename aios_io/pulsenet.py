"""Basic placeholder for PulseNet communication layer."""
import asyncio
from typing import Dict, Tuple


class PulseNet:
    """Minimal peer registry using TCP for demonstration."""

    def __init__(self) -> None:
        self.peers: Dict[str, Tuple[str, int]] = {}

    def register_peer(self, name: str, host: str, port: int) -> None:
        self.peers[name] = (host, port)

    async def start_server(self, host: str, port: int, handler) -> None:
        """Start a simple TCP server and pass incoming messages to handler."""

        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            data = await reader.read(1024)
            if data:
                handler(data.decode())
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(_handle, host, port)
        async with server:
            await server.serve_forever()

    async def send(self, name: str, message: str) -> None:
        if name not in self.peers:
            raise ValueError("Unknown peer")
        host, port = self.peers[name]
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(message.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
