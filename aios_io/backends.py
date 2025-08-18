"""Backend adapters for persisting scheduler state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class BaseBackend:
    """Abstract base backend."""

    def save(self, data: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def load(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class JSONFileBackend(BaseBackend):
    """Persist scheduler state to a JSON file."""

    def __init__(self, path: str = "scheduler_state.json") -> None:
        self.path = Path(path)

    def save(self, data: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)


class RedisBackend(BaseBackend):
    """Store scheduler state in Redis if available."""

    def __init__(self, client: Any, key: str = "scheduler_state") -> None:
        self.client = client
        self.key = key

    def save(self, data: Dict[str, Any]) -> None:
        self.client.set(self.key, json.dumps(data))

    def load(self) -> Dict[str, Any]:
        raw = self.client.get(self.key)
        if not raw:
            return {}
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)


class CeleryBackend(BaseBackend):
    """Persist state using a Celery result backend."""

    def __init__(self, app: Any, key: str = "scheduler_state") -> None:
        self.app = app
        self.key = key

    def save(self, data: Dict[str, Any]) -> None:
        backend = getattr(self.app, "backend", None)
        if backend is None:
            raise RuntimeError("Celery app has no backend")
        backend.set(self.key, json.dumps(data))

    def load(self) -> Dict[str, Any]:
        backend = getattr(self.app, "backend", None)
        if backend is None:
            raise RuntimeError("Celery app has no backend")
        raw = backend.get(self.key)
        if not raw:
            return {}
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
