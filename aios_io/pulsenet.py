"""Asynchronous peer-to-peer messaging layer for AIOS IO."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from pathlib import Path
from typing import Awaitable, Callable, Dict, Tuple


Handler = Callable[[str], Awaitable[None]]

logger = logging.getLogger(__name__)


class PulseNet:
    """Tiny message bus supporting retries, routing and optional TLS."""

    def __init__(
        self,
        identity: str = "local",
        retries: int = 3,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        ca_cert: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self.identity = identity
        self.peers: Dict[str, Tuple[str, int]] = {}
        self.handlers: Dict[str, Handler] = {}
        self.peer_keys: Dict[str, str] = {}
        self.routing_table: Dict[str, str] = {}
        self.retries = retries
        self.config_path = Path(config_path or "pulsenet_peers.json")
        self._load_config()

        self.server_ctx: ssl.SSLContext | None = None
        self.client_ctx: ssl.SSLContext | None = None
        if tls_cert and tls_key:
            self.server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.server_ctx.load_cert_chain(tls_cert, tls_key)
            self.client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            self.client_ctx.load_cert_chain(tls_cert, tls_key)
            if ca_cert:
                self.server_ctx.load_verify_locations(ca_cert)
                self.server_ctx.verify_mode = ssl.CERT_REQUIRED
                self.client_ctx.load_verify_locations(ca_cert)
                self.client_ctx.check_hostname = False

    # ------------------------------------------------------------------
    # Peer management
    def _load_config(self) -> None:
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
            except json.JSONDecodeError:
                data = {}
            for name, info in data.get("peers", {}).items():
                self.peers[name] = (info["host"], info["port"])
                if info.get("key"):
                    self.peer_keys[name] = info["key"]

    def _save_config(self) -> None:
        data = {
            "peers": {
                name: {
                    "host": h,
                    "port": p,
                    "key": self.peer_keys.get(name),
                }
                for name, (h, p) in self.peers.items()
            }
        }
        self.config_path.write_text(json.dumps(data))

    # ------------------------------------------------------------------
    # Peer management
    def register_peer(
        self, name: str, host: str, port: int, key: str | None = None
    ) -> None:
        self.peers[name] = (host, port)
        if key:
            self.peer_keys[name] = key
        self._save_config()

    def set_key(self, name: str, key: str) -> None:
        if name not in self.peers:
            raise ValueError("Unknown peer")
        self.peer_keys[name] = key
        self._save_config()

    def peer_status(self) -> Dict[str, Dict[str, object]]:
        return {
            name: {"host": h, "port": p, "has_key": name in self.peer_keys}
            for name, (h, p) in self.peers.items()
        }

    def register_handler(self, event: str, handler: Handler) -> None:
        """Register an asynchronous handler for an event type."""

        self.handlers[event] = handler

    def register_route(self, msg_type: str, peer_name: str) -> None:
        self.routing_table[msg_type] = peer_name

    async def send_type(self, msg_type: str, event: str, data: str) -> None:
        if msg_type not in self.routing_table:
            raise ValueError("Unknown message type")
        await self.send(
            self.routing_table[msg_type], event, data, msg_type=msg_type
        )

    # ------------------------------------------------------------------
    # Networking primitives
    async def start_server(self, host: str, port: int) -> None:
        """Start a JSON based TCP server and dispatch to handlers."""

        async def _handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            data = await reader.read(65536)
            if data:
                try:
                    message = json.loads(data.decode())
                    peer = message.get("peer")
                    key = message.get("key")
                    if self.peer_keys:
                        expected = self.peer_keys.get(peer)
                        if not peer or not key or expected != key:
                            logger.error("Authentication failed for peer %s", peer)
                            writer.close()
                            await writer.wait_closed()
                            return
                    event = message.get("event")
                    payload = message.get("data")
                    handler = self.handlers.get(event)
                    if handler:
                        await handler(payload)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(
            _handle, host, port, ssl=self.server_ctx
        )
        async with server:
            await server.serve_forever()

    async def _send_bytes(self, host: str, port: int, data: bytes) -> None:
        for attempt in range(self.retries):
            try:
                reader, writer = await asyncio.open_connection(
                    host, port, ssl=self.client_ctx
                )
                writer.write(data)
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                break
            except OSError as exc:
                logger.error(
                    "Send attempt %s to %s:%s failed: %s",
                    attempt + 1,
                    host,
                    port,
                    exc,
                )
                if attempt == self.retries - 1:
                    raise
                await asyncio.sleep(0.1 * 2**attempt)

    async def send(
        self, name: str, event: str, data: str, msg_type: str = "message"
    ) -> None:
        """Send an event to a specific peer."""

        if name not in self.peers:
            raise ValueError("Unknown peer")
        host, port = self.peers[name]
        payload = {
            "type": msg_type,
            "event": event,
            "data": data,
            "peer": self.identity,
        }
        key = self.peer_keys.get(name)
        if key:
            payload["key"] = key
        await self._send_bytes(host, port, json.dumps(payload).encode())

    async def broadcast(
        self, event: str, data: str, msg_type: str = "message"
    ) -> None:
        """Send an event to all known peers."""

        await asyncio.gather(
            *(self.send(name, event, data, msg_type=msg_type) for name in self.peers)
        )
